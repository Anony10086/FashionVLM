import os
from functools import partial
from tqdm import tqdm
import json
import time
import argparse

from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import torch
from torch import tensor
from torch.utils.data import DataLoader

from dataset import FashionRecDatasetBase
from show_o.training.prompting_utils import (UniversalPrompting,
                                             create_attention_mask_for_mmu,
                                             create_attention_mask_predict_next)
from show_o.models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from show_o.training.utils import image_transform


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='fashion_vlm',
                    help='method name, fashion_vlm | show_o')
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
args = parser.parse_args()


if args.method == 'fashion_vlm':
    SYSTEM_PROMPT = ''
else:
    SYSTEM_PROMPT = ('You are a fashion stylist. Recommend an item according to user`s query and uploaded image. '
                     'Recommend only one item for each query.')


class EvalDataset(FashionRecDatasetBase):
    def __getitem__(self, index):
        image = self.images[index]

        json_data = self.jsons[index]
        conversations = json_data["conversation"]
        # query = conversations[0]['value']

        # answer = conversations[1]['value']
        return {
            "key": json_data['key'],
            "image": image,
            "conversations": conversations,
            # "answer": answer,
            # "target_image": json_data['target_items'][0]["path"],
            # "target_category": json_data['target_items'][0]["subcategory"],
            # "target_description": json_data['target_items'][0]["description"]
        }


def collate_fn(batch, uni_prompting=None, img_tokenizer=None):
    device = img_tokenizer.device
    key = [x['key'] for x in batch]

    image = [image_transform(x['image'], resolution=512) for x in batch]
    image = torch.stack(image, dim=0).to(device)
    image_tokens = img_tokenizer.get_code(image) + len(uni_prompting.text_tokenizer)

    prompt_batch = []
    for x in batch:
        conversations = x['conversations']
        prompt = SYSTEM_PROMPT
        for conv_dict in conversations[:-1]:
            role = conv_dict['from']
            message = conv_dict['value']
            if message:
                if role == 'human':
                    prompt += 'USER: ' + message + ' '
                elif role == 'gpt':
                    prompt += 'ASSISTANT: ' + message + "<|endoftext|>"
        prompt += 'ASSISTANT:'
        print(prompt)
        prompt_batch.append(prompt)
        # prompt2 = 'USER: \n' + SYSTEM_PROMPT + x['conversations'][0]['value'] + ' ASSISTANT:'

    # prompt_batch = ['USER: \n' + SYSTEM_PROMPT + x['conversations'][0]['value'] + ' ASSISTANT:' for x in batch]
    text_tokens = uni_prompting.text_tokenizer(
        prompt_batch,
        max_length=381,
        padding=True,
        # padding_side="left",
        truncation=True,
        return_tensors="pt"
    )
    text_tokens = text_tokens['input_ids'].to(device)

    batch_size = text_tokens.shape[0]
    input_ids = torch.cat([
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
        image_tokens,
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
        (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
        text_tokens
    ], dim=1).long()

    attention_mask = create_attention_mask_for_mmu(
        input_ids.to(device),
        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])
    )
    return input_ids, attention_mask, key


#######################################
#############Loading Model#############
#######################################
config = OmegaConf.load(f"./show_o/configs/showo_fashionrec_tuning_3_512x512.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Universal Prompting
tokenizer = AutoTokenizer.from_pretrained(
    config.model.showo.llm_model_path,
    padding_side="left"
)
uni_prompting = UniversalPrompting(
    tokenizer,
    max_text_len=config.dataset.preprocessing.max_seq_length,
    special_tokens=(
        "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
    ),
    ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob
)

# Init VQ model
print("Loading VQ model")
vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
vq_model.requires_grad_(False)
vq_model.eval()

# Loading Fashion VLM
print("Loading Fashion VLM params")
if args.method == "fashion_vlm":
    fashion_vlm = Showo(**config.model.showo).to(device)
    path = os.path.join(config.experiment.output_dir, "pytorch_model.bin")
    print(f"Resuming from checkpoint {path}")
    state_dict = torch.load(path, map_location=device)
    fashion_vlm.load_state_dict(state_dict, strict=False)
elif args.method == "show_o":
    fashion_vlm = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(device)
else:
    raise NotImplementedError(f"Model {args.method} not implemented")
fashion_vlm.eval()

#######################################
##############Loading Data#############
#######################################
data_set_root = f"./datasets/FashionRec/data"

dataset = EvalDataset(
    tar_files=f"{data_set_root}/{args.task}/test/000.tar",
    num_examples=500
)
dataloader = DataLoader(
    dataset,
    batch_size=1,  # must set to 1
    shuffle=False,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    collate_fn=partial(collate_fn, uni_prompting=uni_prompting, img_tokenizer=vq_model)
)


#######################################
#########Strating Inference############
#######################################
# Generation Params
max_new_tokens = 1000
temperature = 0.8
top_k = 1


results = {}
start_time = time.perf_counter()  # 记录总时间
for idx, (input_ids, attention_mask, key) in enumerate(tqdm(dataloader)):
    cont_toks_list = fashion_vlm.mmu_generate(
        input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, top_k=top_k,
        eot_token=uni_prompting.sptids_dict['<|eot|>']
    )
    cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
    text = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)

    sample_id = key[0]
    results[sample_id] = text[0].strip()
    print(f"ASSISTANT: {results[sample_id]}")

end_time = time.perf_counter()  # 记录总结束时间
total_time = end_time - start_time  # 总时间（包括后处理）
print(f"Total Cost Time: {total_time:.2f} seconds")
print(f"Average Batch Latency: {total_time / (len(dataset) + 1):.2f} seconds")

save_dir = f"./output/{args.method}"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/{args.task}_results.json", "w") as f:
    json.dump(results, f, indent=2)
