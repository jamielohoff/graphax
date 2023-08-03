"""
Tools that provides the tools to save a comp. graph repr., meta-data 
and the corresponding source code to a hdf5 file respectively.
Comp. graph repr. are stored in safe-preeliminated form to save memory.
"""
import os
from typing import Sequence, Tuple

import h5py
import jax
import jax.numpy as jnp

from chex import Array


def get_prompt_list(prompt_file: str):
    out = None
    with open(prompt_file) as pf:
        out = pf.readlines()
    prompts = out[0::2]
    make_jaxpr = out[1::2]
    return [(prompt.strip(), mj.strip()) for prompt, mj in zip(prompts, make_jaxpr)]


def check_graph_shape(info: Sequence[int], max_info: Sequence[int]):
    a = info[0] <= max_info[0]
    b = info[1] <= max_info[1]
    c = info[2] <= max_info[2]
    return jnp.logical_and(jnp.logical_and(a, b), c)


def write(fname: str, samples: Tuple[str, Array]):
    assert os.path.isfile(fname) == True
    
    batchsize = len(samples)
    with h5py.File(fname, "a") as file:
        header = file["header"]
        num_samples = header.attrs["num_samples"]
        idx = header.attrs["current_idx"]
        
        if idx + batchsize > num_samples:
            samples = samples[0:num_samples-idx]
            print("Maximum file size reached!")
        
        code_dset = file["data/code"]
        graph_dset = file["data/graph"]
        
        code_dset[idx:idx+batchsize] = [sample[0] for sample in samples]
        
        for i, sample in enumerate(samples):
            edges = sample[1]        
            graph_dset[idx+i:idx+i+1] = edges[None,:,:,:]

        header.attrs["current_idx"] = idx + batchsize
        
        
def create(fname: str, num_samples: int, max_info: Sequence[int] = (20, 105, 20)):
    assert os.path.isfile(fname) == False
    max_i, max_v, max_o = max_info
    
    with h5py.File(fname, "w") as file:
        header = file.create_group("header", (1,))
        header.attrs["num_samples"] = num_samples
        header.attrs["max_graph_shape"] = max_info
        header.attrs["current_idx"] = 0
        
        data = file.create_group("data")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        source_code = file.create_dataset("data/code", (num_samples,), dtype=str_dtype)
        
        graph_dims = (5, max_i+max_v+1, max_v)
        edges = file.create_dataset("data/graph", (num_samples,)+graph_dims, dtype="i4")
    
    
def read(fname: str, batch_idxs: Sequence[int]) -> Tuple[str, Array]:
    """
    Codes variable needs to be decoded using .decode("utf-8")!
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname) as file:        
        codes = file["data/code"][batch_idxs]
        graphs = file["data/graph"][batch_idxs]
        return codes, graphs
    
    
def read_graph(fname: str, batch_idxs: Sequence[int]) -> Tuple[Array]:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname) as file:       
        graphs = file["data/graph"][batch_idxs]
        return graphs
    
    
def read_file_size(fname: str) -> int:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:
        header = file["header"]
        return header.attrs["num_samples"]

