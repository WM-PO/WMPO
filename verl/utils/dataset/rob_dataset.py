# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union
import random
import copy
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
try:
    from libero.libero import benchmark
except:
    print("libero is not installed")
import pickle

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output

class StateDataset(Dataset):
    """
    PyTorch Dataset that loads and returns env states saved with sample_env_states(...).
    """
    def __init__(self, state_path: str):
        with open(state_path, "rb") as f:
            self.states = pickle.load(f)
        for idx, state in enumerate(self.states):
            state['state_id'] = torch.tensor(idx)
            for key, value in state.items():
                if isinstance(value, np.ndarray):
                    state[key] = torch.from_numpy(value)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

class AlohaDataset(Dataset):
    """
    PyTorch Dataset that loads and returns env states saved with sample_env_states(...).
    """
    def __init__(self):
        self.states = []
        for idx in range(128):
            self.states.append({'state_id': torch.tensor(idx), 'states': torch.tensor(-1)})

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return  self.states[idx]


class LIBERO_Dataset(Dataset):
    def __init__(self,
                 task_suite_name,
                 num_trials_per_task=50,
                 task_id = None,
                 train_val ="train",
                 pad_to=0,
                 ):
        
        self.task_suite_name = task_suite_name  
        self.num_trials_per_task = num_trials_per_task  
        self.train_val = train_val
        self.pad_to = pad_to
        self.task_id = task_id
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        if self.task_id is not None:
            task_ids = [self.task_id]
        else:
            task_ids = list(range(task_suite.n_tasks))
        dataframes = []
        
        if self.task_suite_name in ["libero_10", "libero_90", "libero_goal",  "libero_object",  "libero_spatial"]:
            for task_id in task_ids:
                if self.train_val == "train":
                    trials_range = list(range(0, int(self.num_trials_per_task)))
                elif self.train_val == "valid":
                    trials_range = list(range(0, int(self.num_trials_per_task)))  
                else:
                    raise ValueError
                for i in trials_range:
                    data = {
                        "task_suite_name": self.task_suite_name,
                        "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                        "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        'drop': torch.tensor(False).unsqueeze(0)
                    }
                    dataframes.append(data)
            self.dataframe = dataframes

            print(f'dataset len: {len(self.dataframe)}')
        else:
            raise ValueError
        if self.pad_to != 0 and self.pad_to > len(dataframes):
            pad_num = self.pad_to - len(dataframes)
            pad_dataframes = [copy.deepcopy(p) for p in random.sample(dataframes, pad_num)]
            for p in pad_dataframes:
                p['drop'] = torch.tensor(True).unsqueeze(0)
            self.dataframe.extend(pad_dataframes)
     

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        return self.dataframe[item]



class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        """Reset for new epoch"""
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.start_new_epoch()
            return next(self.dataloader_iter)
            # raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)
