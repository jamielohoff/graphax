from typing import Dict, Sequence, Tuple
import os
from torch.utils.data import Dataset

import jax 
import jax.numpy as jnp

import chex

from graphax.dataset.utils import read, read_graph_info, read_file_size


class GraphDataset(Dataset):
    files: Sequence[str]
    file_sizes: Sequence[int]
    length: int
    
    def __init__(self, dir: str) -> None:
        self.length = 0
        self.files, self.file_sizes = [], []
        for file in os.listdir(dir):
            if file.endswith(".hdf5"):
                path = os.path.join(dir, file)
                self.files.append(path)
                file_size = read_file_size(path)
                self.file_sizes.append(file_size)
                self.length += file_size
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[chex.Array, chex.Array]:
        file_idx = [fs for fs in self.file_sizes if fs < idx]
        file = self.files[len(file_idx)-1]
        _idx = idx - sum(file_idx)
        graph, info = read_graph_info(file, _idx)
        return graph, info

dataset = GraphDataset("./tests")
print(len(dataset))
print(dataset[99])

