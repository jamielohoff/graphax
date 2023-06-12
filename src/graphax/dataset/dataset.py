from typing import Dict, Sequence, Tuple
import os
from torch.utils.data import Dataset

import chex

from .utils import read, read_graph_info, read_file_size


class GraphDataset(Dataset):
    files: Sequence[str]
    file_sizes: Sequence[int]
    length: int
    include_code: bool
    
    def __init__(self, dir: str, include_code: bool = False) -> None:
        self.length = 0
        self.include_code = include_code
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
        file_idx = [fs for i, fs in enumerate(self.file_sizes) if sum(self.file_sizes[:i+1]) <= idx]
        file = self.files[len(file_idx)-1]
        _idx = idx - sum(file_idx)
        
        if self.include_code:
            src, graph, info, vertices, attn_mask = read(file, _idx)
            return src, graph, info, vertices, attn_mask
        else:
            graph, info, vertices, attn_mask = read_graph_info(file, _idx)
            return graph, info, vertices, attn_mask

