import os
import sys
sys.path.append(os.getcwd())
from functools import partial
from tqdm import tqdm
import argparse
import time

from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import torch
import numpy as np
from torch.utils.data import DataLoader
import ollama

from dataset import FashionImageGenDatasetBase
from show_o.training.prompting_utils import (UniversalPrompting,
                                             create_attention_mask_predict_next)
from show_o.models import Showo, MAGVITv2, get_mask_chedule


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='fashion_vlm',
                    help='method name, fashion_vlm | show_o | show_o_ablation | gpt_4o | gpt_4.1 | llama_32 | llava_16 | llava_onevision_05b_si | llava_onevision_7b_ov_chat')
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
args = parser.parse_args()


def collate_fn(batch, uni_prompting=None):
    ids = [x[0] for x in batch]
    batch_prompt = []
    for x in batch:
        prompt = x[1]
        messages = [
            {
                'role': 'system',
                'content': 'You need to extract the recommended item description from the user`s message. '
                           'If there is no recommended item, just output None.'
                           'Summarize the description of the recommended item. You must classify what is provided and what is recommended.'
                           'We only need the description of the recommended items.'
                           'IMPORTANT: Reply in One scentence. Use two examples I provided',
            },
            {
                'role': 'user',
                'content': "It sounds like you're aiming for a chic and coordinated look! Based on your preference for knee-high styles with buckles, I recommend checking out some dark brown knee-high boots featuring a buckle detail at the ankle. They would beautifully complement your outfits and add a touch of sophistication. Let me know if that fits what you're looking for!",
            },
            {
                'role': 'assistant',
                'content': "dark brown knee-high boots featuring a buckle detail at the ankle"
            },
            {
                'role': 'user',
                'content': "Yes, I recommend the following sandals:",
            },
            {
                'role': 'assistant',
                'content': "Sandal"
            },
            {
                'role': 'user',
                'content': prompt,
            }
        ]
        response = ollama.chat(
            model='phi4',
            messages=messages
        )
        text = response['message']['content']
        print(f"{x[0]}: \nOriginal text: {prompt}\nExtracted text: {text}")
        batch_prompt.append(text)

    # 1024 is num_vq_tokens of show-o
    mask_token_id = 58497
    image_tokens = torch.ones((len(batch_prompt), 1024), dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting((batch_prompt, image_tokens), 't2i_gen')
    uncond_input_ids, _ = uni_prompting(([''] * len(batch_prompt), image_tokens), 't2i_gen')
    attention_mask = create_attention_mask_predict_next(
        torch.cat([input_ids, uncond_input_ids], dim=0),
        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
        rm_pad_in_image=True
    )
    return ids, input_ids, uncond_input_ids, attention_mask


#######################################
#############Loading Model#############
#######################################
config = OmegaConf.load(f"./show_o/outputs/FashionVLM-2025-03-30/config_infer.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Init Universal Prompting
tokenizer = AutoTokenizer.from_pretrained(
    config.model.showo.llm_model_path,
    padding_side="left",
    local_files_only=True
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
vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name, local_files_only=True).to(device)
vq_model.requires_grad_(False)
vq_model.eval()

# Loading Fashion VLM
print("Loading Image Generation Model params")
if args.method in ["show_o", "show_o_ablation"]:
    fashion_vlm = Showo.from_pretrained(config.model.showo.pretrained_model_path, local_files_only=True).to(device)
else:
    fashion_vlm = Showo(**config.model.showo).to(device)
    path = os.path.join(config.experiment.output_dir, "pytorch_model.bin")
    print(f"Resuming from checkpoint {path}")
    state_dict = torch.load(path, map_location=device)
    fashion_vlm.load_state_dict(state_dict, strict=False)
fashion_vlm.eval()

#######################################
##############Loading Data#############
#######################################
result_dir = f"./output/{args.method}"
if args.method in ['gpt_4o', 'o3_mini', 'o4_mini', 'gpt_4.1']:
    json_file_name = f"{args.task}_results.jsonl"
else:
    json_file_name = f"{args.task}_results.json"

dataset = FashionImageGenDatasetBase(
    json_file_path=f"{result_dir}/{json_file_name}",
    num_examples=500
)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    collate_fn=partial(collate_fn, uni_prompting=uni_prompting)
)


#######################################
#########Strating Inference############
#######################################
image_save_dir = os.path.join(result_dir, args.task)
os.makedirs(image_save_dir, exist_ok=True)
start_time = time.perf_counter()  # 记录总时间
for ids, input_ids, uncond_input_ids, attention_mask in tqdm(dataloader):
    input_ids = input_ids.to(device)
    uncond_input_ids = uncond_input_ids.to(device)
    attention_mask = attention_mask.to(device)

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    with torch.no_grad():
        gen_token_ids = fashion_vlm.t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=5,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=50,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )

    gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    for index, images in zip(ids, images):
        image = Image.fromarray(images)
        image.save(os.path.join(image_save_dir, f"{index}_gen.jpg"))

end_time = time.perf_counter()  # 记录总结束时间
total_time = end_time - start_time  # 总时间（包括后处理）
print(f"Total Cost Time: {total_time:.2f} seconds")
print(f"Average Batch Latency: {total_time / (len(dataset) + 1):.2f} seconds")
