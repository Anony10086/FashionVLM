import os
import json
import tarfile

from PIL import Image
from torch.utils.data import Dataset
from braceexpand import braceexpand
from omegaconf.listconfig import ListConfig


class FashionRecDatasetBase(Dataset):
    """A dataset class for outfit completion using WebDataset.

    This dataset loads data from tar files where each sample contains:
    - A JSON file with conversation and outfit data (partial outfit and target items)
    - An image file (merged 4-grid image of the partial outfit)

    Attributes:
        tar_files (list): List of paths to tar files containing the dataset
    """
    def __init__(
        self,
        tar_files,
        num_examples=10000
    ):
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

        self._load_tar_files(train_shards_path_or_url[0], num_examples)

    def _load_tar_files(self, tar_path: str, num_examples: int):
        self.images = []
        self.jsons = []
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Extract the base name (e.g., "000" from "000.jpg")
                    filename = os.path.basename(member.name)  # get 0009741.jpg
                    base_name = os.path.splitext(filename)[0]  # get 0009741

                    if int(base_name) > num_examples - 1:
                        break

                    if member.name.endswith(".jpg"):
                        with tar.extractfile(member) as jpg_file:
                            self.images.append(Image.open(jpg_file).convert("RGB"))
                    elif member.name.endswith(".json"):
                        with tar.extractfile(member) as json_file:
                            self.jsons.append(json.loads(json_file.read().decode("utf-8")))

    def __len__(self):
        return len(self.jsons)

    def __getitem__(self, index):
        image = self.images[index]
        json_data = self.jsons[index]
        return image, json_data


class FashionImageGenDatasetBase(Dataset):
    def __init__(self, json_file_path, num_examples=10000):
        self.results = []
        if json_file_path.endswith('json'):
            with open(json_file_path, "r") as f:
                json_data = json.load(f)
            self.results = [(k, v) for k, v in json_data.items()]
        elif json_file_path.endswith('jsonl'):
            with open(json_file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    index = data['custom_id'].split('-')[-1]
                    response = data['response']['body']['choices'][0]['message']['content']
                    self.results.append((index, response))
        else:
            raise ValueError("json_file_path must be a json or jsonl file")

        self.results = self.results[:num_examples]

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]


if __name__ == '__main__':
    data_set_root = "/mnt/d/PostDoc/fifth paper/code/FashionVLM/datasets/FashionRec/data"
    task = "basic_recommendation"  # basic_recommendation | personalized_recommendation | alternative_recommendation
    dataset = FashionRecDatasetBase(
        tar_files=f"{data_set_root}/{task}/test/000.tar",
        num_examples=10
    )
    x = dataset[0]
