import os
from functools import partial
from tqdm import tqdm
import json
import time

from PIL import Image
from omegaconf import OmegaConf
from transformers import AutoTokenizer
import torch
from torch import tensor
from torch.utils.data import DataLoader, Dataset
import pandas as pd

from dataset import FashionRecDatasetBase
from show_o.training.prompting_utils import (UniversalPrompting,
                                             create_attention_mask_for_mmu,
                                             create_attention_mask_predict_next)
from show_o.models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from show_o.training.utils import image_transform


SYSTEM_PROMPT = "You are a fashion stylist. Recommend an item according to user`s query and uploaded image. Recommend only one item for each query."


class EvalDataset(Dataset):
    def __init__(self, dataset_root, task, num_examples=10000):
        self.items = pd.read_parquet(os.path.join(dataset_root, 'meta', 'items_lite.parquet'))
        self.samples = []
        for i in range(num_examples):
            index = f'{i:07d}'
            with open(os.path.join(dataset_root, 'data', task, 'test/temp', f'{index}.json'), 'r') as f:
                self.samples.append(json.load(f))

    def __getitem__(self, index):
        image = self.images[index]

        json_data = self.jsons[index]
        conversations = json_data["conversation"]
        query = conversations[0]['value']

        answer = conversations[1]['value']
        return {
            "key": json_data['key'],
            "image": image,
            "query": query,
            "answer": answer,
            "target_image": json_data['target_items'][0]["path"],
            "target_category": json_data['target_items'][0]["subcategory"],
            "target_description": json_data['target_items'][0]["description"]
        }

    def __len__(self):
        return len(self.samples)


# def collate_fn(batch, uni_prompting=None, img_tokenizer=None):
#     device = img_tokenizer.device
#     image = [image_transform(x['image'], resolution=512) for x in batch]
#     image = torch.stack(image, dim=0).to(device)
#     image_tokens = img_tokenizer.get_code(image) + len(uni_prompting.text_tokenizer)
#
#     query = ['USER: \n' + SYSTEM_PROMPT + x['query'] + ' ASSISTANT:' for x in batch]
#     text_tokens = uni_prompting.text_tokenizer(
#         query,
#         max_length=381,
#         padding=True,
#         # padding_side="left",
#         truncation=True,
#         return_tensors="pt"
#     )
#     text_tokens = text_tokens['input_ids'].to(device)
#
#     batch_size = text_tokens.shape[0]
#     input_ids = torch.cat([
#         (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
#         (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
#         image_tokens,
#         (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
#         (torch.ones(batch_size, 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
#         text_tokens
#     ], dim=1).long()
#
#     attention_mask = create_attention_mask_for_mmu(
#         input_ids.to(device),
#         eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])
#     )
#     return input_ids, attention_mask


#######################################
#############Loading Model#############
#######################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#######################################
##############Loading Data#############
#######################################
data_set_root = "/mnt/d/PostDoc/fifth paper/code/FashionVLM/datasets/FashionRec"
task = "basic_recommendation"  # basic_recommendation | personalized_recommendation | alternative_recommendation
dataset = EvalDataset(
    dataset_root=data_set_root,
    task=task,
    num_examples=500
)
dataloader = DataLoader(
    dataset,
    batch_size=1,  # must set to 1
    shuffle=False,
    pin_memory=False,
    num_workers=0,
    drop_last=False,
    # collate_fn=partial(collate_fn, uni_prompting=uni_prompting, img_tokenizer=vq_model)
)


#######################################
#########Strating Inference############
#######################################
# Generation Params


results = {}
start_time = time.perf_counter()  # 记录总时间
for idx, (input_ids, attention_mask) in enumerate(tqdm(dataloader)):
    a = 1

end_time = time.perf_counter()  # 记录总结束时间
total_time = end_time - start_time  # 总时间（包括后处理）
print(f"Total Cost Time: {total_time:.2f} seconds")
print(f"Average Batch Latency: {total_time / (len(dataset) + 1):.2f} seconds")

save_dir = f"/mnt/d/PostDoc/fifth paper/code/FashionVLM/output/difashion/{task}"

