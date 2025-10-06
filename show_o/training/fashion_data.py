import os
import json
from functools import partial
import math

import numpy as np
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import default_collate
from PIL import Image
import torch
from omegaconf import ListConfig
from braceexpand import braceexpand
import webdataset as wds
from omegaconf.listconfig import ListConfig

from training.utils import image_transform, merge_images
from llava import conversation as conversation_lib


DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100
conversation_lib.default_conversation = conversation_lib.conv_templates["phi1.5"]
"""
conv_phi_v0 = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v0",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)
"""
SYSTEM_PROMPT = ""


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def is_valid_conversation(conversation):
    # 检查是否是列表
    if not isinstance(conversation, list):
        return False

    # 检查列表中每个元素是否是字典
    for item in conversation:
        if not isinstance(item, dict):
            return False
        # 检查字典中是否有必要的键
        if "from" not in item or "value" not in item:
            return False

        if not isinstance(item["from"], str):
            return False

        if not isinstance(item["value"], str):
            return False

    return True


def preprocess_v0(
    sources,
    tokenizer,
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )


class FashionRecDataset:
    """A dataset class for outfit completion using WebDataset.

    This dataset loads data from tar files where each sample contains:
    - A JSON file with conversation and outfit data (partial outfit and target items)
    - An image file (merged 4-grid image of the partial outfit)

    Attributes:
        tar_files (list): List of paths to tar files containing the dataset
        tokenizer: Tokenizer for processing text descriptions
        resolution (int): Resolution for image resizing
    """
    def __init__(
        self,
        tar_files,
        tokenizer,
        batch_size=5,
        global_batch_size=5,
        num_workers=1,
        max_length=381,
        pin_memory=True,
        persistent_workers=True,
        resolution=512,
        samples_per_tar: int = 10000,
        shuffle_buffer: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.samples_per_tar = samples_per_tar  # 每个 tar 文件的样本数（之前设定为 10000）
        self.shuffle_buffer = shuffle_buffer  # shuffle 缓冲区大小

        # 处理 tar_files（可能是字符串或列表）
        if isinstance(tar_files, str):
            train_shards_path_or_url = list(braceexpand(tar_files))
        elif isinstance(tar_files, (list, tuple, ListConfig)):
            if isinstance(tar_files, ListConfig):
                tar_files = list(tar_files)
            train_shards_path_or_url = []
            for tar_pattern in tar_files:
                expanded_files = list(braceexpand(tar_pattern))
                train_shards_path_or_url.extend(expanded_files)
        else:
            raise ValueError("tar_files must be a string or a list of strings")

        # 验证 tar 文件是否存在
        for tar_file in train_shards_path_or_url:
            if not os.path.exists(tar_file):
                raise FileNotFoundError(f"Tar file {tar_file} does not exist")

        # 定义 processing pipeline
        processing_pipeline = [
            wds.decode("pil"),  # 解码图片
            # wds.map(self.load_json),
            wds.rename(
                images="jpg",  # 重命名图片键
                text="json",
            ),
            wds.map(self.process_json),  # 处理 JSON 数据
            wds.map_dict(images=partial(image_transform, resolution=resolution)),  # 图像预处理
            wds.map(filter_keys(["images", "input_ids", "labels", "input_ids_system"])),
        ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            wds.tarfile_to_samples(),
            wds.shuffle(self.shuffle_buffer),
            *processing_pipeline,
            wds.batched(batch_size, partial=True, collation_fn=partial(self.collate_fn, tokenizer=self.tokenizer, max_length=max_length)),
        ]

        num_train_examples = len(train_shards_path_or_url) * samples_per_tar
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers))  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def process_json(self, sample):
        """Extract conversations from JSON data."""
        conversation = sample['text']['conversation']
        if not conversation or not is_valid_conversation(conversation):
            return None
        data_dict = preprocess_v0([conversation], self.tokenizer)
        sample['input_ids'] = data_dict['input_ids'][0]
        sample['labels'] = data_dict['labels'][0]
        sample['input_ids_system'] = data_dict['input_ids_system'][0]
        return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @staticmethod
    def collate_fn(
            instances,
            tokenizer=None,
            max_length=77,
    ):
        input_ids, labels, input_ids_system = tuple([instance[key] for instance in instances]
                                                    for key in ("input_ids", "labels", "input_ids_system"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids_system = torch.stack(input_ids_system, dim=0)

        offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

        if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
            pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, pad_tube], dim=1)

            pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
            labels = torch.cat([labels, pad_tube], dim=1)

        min_max_len = min(
            max_length - input_ids_system.shape[-1],
            tokenizer.model_max_length - input_ids_system.shape[-1],
        )

        input_ids = input_ids[:, :min_max_len]
        labels = labels[:, :min_max_len]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            input_ids_system=input_ids_system,
        )

        images = [instance['images'] for instance in instances]
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images, dim=0)

        return batch


if __name__ == '__main__':
    import transformers
    pretrained_model_path = "microsoft/phi-1_5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_path,
                                                           padding_side="left")
    special_tokens = ("soi", "eoi", "sovi", "eovi", "t2i", "mmu", "t2v", "v2v", "lvg")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_tokens(list(special_tokens))
    train_dataset = FashionRecDataset(
        tar_files=["/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/FashionRec/data/basic_recommendation/{000..001}.tar",
        "/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/FashionRec/data/personalized_recommendation/{000..001}.tar",
        "/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/FashionRec/data/alternative_recommendation/000.tar"],
        tokenizer=tokenizer,
        resolution=512,
        max_length=381,
        samples_per_tar=10000,
        shuffle_buffer=1000
    )
    dataloader = train_dataset.train_dataloader
    from torchvision.transforms import functional as f
    # for x in dataloader:
    #     f.to_pil_image(x['images'][0]).save('tmp.jpg')
    a = next(iter(dataloader))
