"""
Tools that provides the tools to save a comp. graph repr., meta-data 
and the corresponding source code to a hdf5 file respectively.
Comp. graph repr. are stored in safe-preeliminated form to save memory.
"""
from typing import Sequence, Tuple

import os
import h5py
import jax
import jax.numpy as jnp
import numpy as np

import chex

from ..core import GraphInfo, make_graph_info
from ..transforms.embedding import embed


def get_prompt_list(prompt_file: str):
    out = None
    with open(prompt_file) as pf:
        out = pf.readlines()
    prompts = out[0::2]
    make_jaxpr = out[1::2]
    return [(prompt.strip(), mj.strip()) for prompt, mj in zip(prompts, make_jaxpr)]


def check_graph_shape(info: GraphInfo, max_graph_shape: Tuple[int, int, int]):
    a = info.num_inputs <= max_graph_shape[0]
    b = info.num_intermediates <= max_graph_shape[1]
    c = info.num_outputs <= max_graph_shape[2]
    return jnp.logical_and(jnp.logical_and(a, b), c)


def write(fname: str, 
        samples: Tuple[str, chex.Array, GraphInfo]) -> None:
    assert os.path.isfile(fname) == True
    
    batchsize = len(samples)
    with h5py.File(fname, "a") as file:
        header = file["header"]
        num_samples = header.attrs["num_samples"]
        graph_shape = header.attrs["max_graph_shape"]
        idx = header.attrs["current_idx"]
        new_graph_info = make_graph_info(graph_shape)
        
        if idx + batchsize > num_samples:
            samples = samples[0:num_samples-idx]
            print("Maximum file size reached!")
        
        code_dset = file["data/code"]
        graph_dset = file["data/graph"]
        info_dset = file["data/info"]
        
        code_dset[idx:idx+batchsize] = [sample[0] for sample in samples]
        
        for i, sample in enumerate(samples):
            arr = embed(sample[1], sample[2], new_graph_info)[0]
            arr = np.array(arr, dtype=np.int16)
            graph_dset[idx+i:idx+i+1] = arr[None,:,:]
        
        info_dset[idx:idx+batchsize, :] = [sample[2] for sample in samples]
        header.attrs["current_idx"] = idx + batchsize
        
        
def create(fname: str, 
            num_samples: int = 3, 
            max_graph_shape: Tuple[int, int, int] = (10, 15, 5)) -> None:
    assert os.path.isfile(fname) == False
    
    with h5py.File(fname, "w") as file:
        header = file.create_group("header", (1,))
        header.attrs["num_samples"] = num_samples
        header.attrs["max_graph_shape"] = max_graph_shape
        header.attrs["current_idx"] = 0
        
        data = file.create_group("data")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        source_code = file.create_dataset("data/code", (num_samples,), dtype=str_dtype)
        
        graph_dims = (max_graph_shape[0]+max_graph_shape[1], max_graph_shape[1]+max_graph_shape[2])
        comp_graph = file.create_dataset("data/graph", (num_samples,)+graph_dims, dtype="i2")
        
        meta_info = file.create_dataset("data/info", (num_samples, 4), dtype="i4")
    
    
def read(fname: str, 
        batch_idxs: Sequence[int]) -> Tuple[str, chex.Array, GraphInfo]:
    """
    codes variable needs to be decoded using .decode("utf-8)!
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:        
        codes = file["data/code"][batch_idxs]
        graphs = file["data/graph"][batch_idxs]
        info = file["data/info"][batch_idxs]
        return codes, graphs, info
    
    
def read_graph_info(fname: str, 
                    batch_idxs: Sequence[int]) -> Tuple[str, chex.Array, GraphInfo]:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:       
        graphs = file["data/graph"][batch_idxs]
        info = file["data/info"][batch_idxs]
        return graphs, info
    
    
def read_file_size(fname: str) -> int:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:
        header = file["header"]
        return header.attrs["num_samples"]

