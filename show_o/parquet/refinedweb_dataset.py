# coding=utf-8
# Copyright 2024 NUS Show Lab.
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

import collections
import random

import torch
import glob
from torch.utils.data import IterableDataset
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import pyarrow.parquet as pq

# from parquet.parquet_dataset import CruiseParquetDataset


class CruiseParquetDataset(IterableDataset):
    def __init__(self,
                 data_path,
                 rank=0,
                 world_size=1,
                 shuffle=True,
                 repeat=True,
                 verbose=True,
                 buffer_size=1000,
                 meta_data_path=None,
                 state_path=None,
                 num_workers=1):
        super().__init__()

        # 保存初始化参数
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.verbose = verbose
        self.buffer_size = buffer_size
        self.num_workers = num_workers

        # 获取所有parquet文件路径
        if isinstance(data_path, str):
            self.parquet_files = sorted(glob.glob(data_path))
        else:
            self.parquet_files = sorted(data_path)

        # 数据缓冲区
        self.buffer_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    def _load_parquet_file(self, file_path):
        """加载单个parquet文件"""
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df.to_dict('records')
        except Exception as e:
            if self.verbose:
                print(f"Error loading {file_path}: {e}")
            return []

    def _buffer_data(self):
        """填充数据缓冲区"""
        while not self.stop_event.is_set():
            if self.shuffle:
                random.shuffle(self.parquet_files)

            for file_path in self.parquet_files:
                if self.stop_event.is_set():
                    break

                records = self._load_parquet_file(file_path)
                if self.shuffle:
                    random.shuffle(records)

                for record in records:
                    if self.stop_event.is_set():
                        break
                    self.buffer_queue.put(record)

            if not self.repeat:
                break

    def generate(self):
        """生成数据的迭代器"""
        buffer_thread = threading.Thread(target=self._buffer_data)
        buffer_thread.start()

        try:
            while True:
                if self.buffer_queue.empty() and not self.repeat:
                    break

                try:
                    data = self.buffer_queue.get(timeout=1)
                    worker_hash = hash(threading.current_thread().name)
                    data_idx = 0  # 这里可以添加具体的索引逻辑
                    seed = random.randint(0, 2 ** 32 - 1)
                    yield data, worker_hash, data_idx, seed
                except queue.Empty:
                    continue

        finally:
            self.stop_event.set()
            buffer_thread.join()

    def __iter__(self):
        return self.generate()

    def __del__(self):
        """清理资源"""
        self.stop_event.set()
        self.executor.shutdown(wait=False)


class RefinedWebDataset(CruiseParquetDataset):
    def __init__(
        self,
        data_path,
        rank: int = 0,
        world_size: int = 1,
        shuffle=True,
        repeat=True,
        buffer_size=1000,
        max_length=8000,
        num_workers=1,
        **kwargs
    ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=False, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)
        self.max_length = max_length

    def __iter__(self):
        for example in self.generate():
            try:
                data, current_worker_hash, data_idx, seed = example
                text = data['content'].replace('\n', '')
                if len(text) > self.max_length:
                    start_index = random.randint(0, len(text) - self.max_length - 1)
                    selected_text = text[start_index:start_index + self.max_length]
                else:
                    selected_text = text
                ret = {'input_ids': selected_text}
                yield ret

            except Exception as e:
                # print('internal dataset iter error', e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

if __name__ == '__main__':

    dataset = RefinedWebDataset('/mnt/d/PostDoc/fifth paper/related work/DiFashion/datasets/falcon-refinedweb/data/*.parquet', num_workers=10)
    a = next(iter(dataset))
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10,
                                  sampler=None, collate_fn=dataset.collate_fn,
                                  num_workers=10)
                                  # num_workers=0)
    for i, batch in enumerate(train_dataloader):
        print(len(batch['input_ids'][0]))
        import ipdb; ipdb.set_trace()
