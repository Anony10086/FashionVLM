# coding=utf-8
# Copyright 2024 The HuggingFace, NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is heavily inspired by https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py

import itertools
import json
import math
import os
import random
import re
from functools import partial
from typing import List, Optional, Union

from PIL import Image

Image.warnings.simplefilter('error', Image.DecompressionBombWarning)

import webdataset as wds
import yaml
from braceexpand import braceexpand
from torch.utils.data import default_collate
from torchvision import transforms
from torchvision.transforms import functional as F
from transformers import PreTrainedTokenizer
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

person_token = ["a person", "someone", "somebody"]


class ResizeLongEdge:
    def __init__(self, size, interpolation=transforms.InterpolationMode.BICUBIC):
        """
        自定义变换：将图像的最长边调整到指定大小，保持宽高比。

        Args:
            size (int): 目标最长边大小（例如 512）。
            interpolation: 插值方法，例如 BICUBIC。
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        """
        调整图像大小，使最长边等于 self.size。

        Args:
            image: PIL 图像。

        Returns:
            调整大小后的 PIL 图像。
        """
        width, height = image.size
        max_side = max(width, height)

        # 如果最长边已经等于目标大小，直接返回
        if max_side == self.size:
            return image

        # 计算缩放比例
        scale = self.size / max_side

        # 计算新的宽高
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 调整大小
        return transforms.Resize(
            (new_height, new_width),
            interpolation=self.interpolation
        )(image)


class CenterCropWithCustomPadding:
    """
    Crops the given image at the center with custom padding value.
    If the image is smaller than the output size, it is padded with the specified fill value.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an int,
            a square crop (size, size) is made. If provided a sequence of length 1,
            it will be interpreted as (size[0], size[0]).
        fill (int or tuple): Padding value for PIL images (default: 256).
            For tensors, this should be a float value after ToTensor.
    """

    def __init__(self, size, fill=256):
        super().__init__()
        self.size = size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        # 获取目标尺寸
        crop_height, crop_width = self.size

        img_width, img_height = img.size

        # 如果图像尺寸小于目标尺寸，填充
        if img_width < crop_width or img_height < crop_height:
            padding_left = max((crop_width - img_width) // 2, 0)
            padding_right = max(crop_width - img_width - padding_left, 0)
            padding_top = max((crop_height - img_height) // 2, 0)
            padding_bottom = max(crop_height - img_height - padding_top, 0)
            img = transforms.Pad(
                padding=(padding_left, padding_top, padding_right, padding_bottom),
                fill=self.fill,
                padding_mode='constant'
            )(img)

        # 中心裁剪
        return F.center_crop(img, self.size)


def replace_person_token(t):
    "Used for CC12M"
    t = re.sub("<person>([,\s]*(and)*[,\s]*<person>)+", " people ", t)
    while "<person>" in t:
        t = t.replace("<person>", f" {random.choices(person_token)} ", 1)
    return t


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=wds.warn_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def image_transform(sample, resolution=256):
    image = sample["images"]
    image = ResizeLongEdge(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = CenterCropWithCustomPadding((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    sample["images"] = image
    return sample


def remove_prefix(caption):
    caption = caption.replace('The image features ', '').replace('The image presents ', '').replace(
        "The image you've sent is, ", '').replace("In the center of the image, ", '').replace(
        "The image showcases ", '').replace("The image is ", '').replace(
        "The image captures ", '').replace("In the given image ", '').replace(
        "The image portrays ", '').replace("In the image, ", '').replace("In this image, we see ", '').replace(
        "The image depicts ", '').replace("This is ", '').replace("In this image, ", '').replace(
        "This image captures ", '')

    return caption


class Text2ImageDataset:
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            tokenizer: PreTrainedTokenizer,
            max_seq_length: int,
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            resolution: int = 256,
            shuffle_buffer_size: int = 1000,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            external_journeydb_caption_path: Optional[str] = '',
            is_captioning: bool = False,
            add_caption_prompt: bool = False,
    ):
        self.external_journeydb_caption_path = external_journeydb_caption_path
        self.is_captioning = is_captioning
        self.add_caption_prompt = add_caption_prompt
        if self.add_caption_prompt:
            with open("./training/questions.json") as f:
                self.caption_prompt = json.load(f)
                self.caption_prompt = ['USER: \n' + prompt + ' ASSISTANT:' for prompt in self.caption_prompt]
        else:
            self.caption_prompt = None

        if external_journeydb_caption_path != '':
            with open(external_journeydb_caption_path) as file:
                self.journeydb_caption = json.load(file)
        else:
            self.journeydb_caption = None

        def tokenize(text):
            if tokenizer is not None:
                text = replace_person_token(text)
                input_ids = tokenizer(
                    text, max_length=max_seq_length, padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                return input_ids[0]
            else:
                return text

        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [list(braceexpand(urls)) for urls in train_shards_path_or_url]
            # flatten list using itertools
            train_shards_path_or_url = list(itertools.chain.from_iterable(train_shards_path_or_url))

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.map(self.load_external_caption, handler=wds.ignore_and_continue),
            wds.rename(
                images="jpg;png;jpeg;webp",
                input_ids="text;txt;caption",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys(set(["images", "input_ids"]))),
            wds.map(partial(image_transform, resolution=resolution), handler=wds.warn_and_continue),
            wds.map_dict(
                input_ids=tokenize,
                handler=wds.warn_and_continue,
            ),
        ]

        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

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
            # num_workers=0,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            # persistent_workers=0,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    def load_external_caption(self, sample):

        if 'txt' not in sample.keys():
            sample['txt'] = ''

        elif self.journeydb_caption is not None and sample['__key__'] in self.journeydb_caption:
            sample['txt'] = random.sample(self.journeydb_caption[sample['__key__']], 1)[0]
            return sample

        else:
            return sample

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


if __name__ == '__main__':
    dataset = Text2ImageDataset(
        train_shards_path_or_url=["/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/FashionRec/data/fashion_image_generation/{000..001}.tar",],
        tokenizer=None,  # we want to get raw texts
        max_seq_length=381,
        num_train_examples=10000,
        per_gpu_batch_size=5,
        global_batch_size=5,
        num_workers=1,
        resolution=512,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=True,
        external_journeydb_caption_path="/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/JourneyDB/data/train/train_journeydb_anno.json",
    )
    train_dataloader_t2i = dataset.train_dataloader
    a = next(iter(train_dataloader_t2i))
    from torchvision.transforms import functional as f
    for i in range(5):
        f.to_pil_image(a['images'][i]).save(f'tmp{i}.jpg')
    a = 1
