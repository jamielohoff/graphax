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


def get_prompt_list(prompt_file: str):
    out = None
    with open(prompt_file) as pf:
        out = pf.readlines()
    prompts = out[0::2]
    make_jaxpr = out[1::2]
    return [(prompt.strip(), mj.strip()) for prompt, mj in zip(prompts, make_jaxpr)]


def check_graph_shape(info: GraphInfo, max_info: GraphInfo):
    a = info.num_inputs <= max_info.num_inputs
    b = info.num_intermediates <= max_info.num_intermediates
    c = info.num_outputs <= max_info.num_outputs
    return jnp.logical_and(jnp.logical_and(a, b), c)


def write(fname: str, 
        samples: Tuple[str, chex.Array, GraphInfo]) -> None:
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
        info_dset = file["data/info"]
        vertices_dset = file["data/vertices"]
        attn_mask_dset = file["data/attn_mask"]
        
        code_dset[idx:idx+batchsize] = [sample[0] for sample in samples]
        
        for i, sample in enumerate(samples):
            edges = np.array(sample[1], dtype=bool)
            vertices = np.array(sample[3], dtype=np.int32)
            attn_mask = np.array(sample[4], dtype=bool)
            
            graph_dset[idx+i:idx+i+1] = edges[None,:,:]
            info_dset[idx+i:idx+i+1, :] = [*sample[2]]
            vertices_dset[idx+i:idx+i+1, :] = vertices
            attn_mask_dset[idx+i:idx+i+1] = attn_mask[None,:, :]

        header.attrs["current_idx"] = idx + batchsize
        
        
def create(fname: str, 
            num_samples: int = 3, 
            max_info: GraphInfo = make_graph_info((10, 15, 5))) -> None:
    assert os.path.isfile(fname) == False
    
    max_i = max_info.num_inputs
    max_v = max_info.num_intermediates
    max_o = max_info.num_outputs
    
    with h5py.File(fname, "w") as file:
        header = file.create_group("header", (1,))
        header.attrs["num_samples"] = num_samples
        header.attrs["max_graph_shape"] = [max_i, max_v, max_o]
        header.attrs["current_idx"] = 0
        
        data = file.create_group("data")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        source_code = file.create_dataset("data/code", (num_samples,), dtype=str_dtype)
        
        graph_dims = (max_i+max_v, max_v+max_o)
        mask_dims = (max_v, max_v)
        edges = file.create_dataset("data/graph", (num_samples,)+graph_dims, dtype=bool)
        meta_info = file.create_dataset("data/info", (num_samples, 4), dtype="i4")
        vertices = file.create_dataset("data/vertices", (num_samples, max_v), dtype="i4")
        attn_mask = file.create_dataset("data/attn_mask", (num_samples,)+mask_dims, dtype=bool)
    
    
def read(fname: str, batch_idxs: Sequence[int]) -> Tuple[str, chex.Array, GraphInfo]:
    """
    codes variable needs to be decoded using .decode("utf-8)!
    """
    assert os.path.isfile(fname) == True
    
    codes, graphs, info = None, None, None
    with h5py.File(fname) as file:        
        codes = file["data/code"][batch_idxs]
        graphs = file["data/graph"][batch_idxs]
        info = file["data/info"][batch_idxs]
        vertices = file["data/vertices"][batch_idxs]
        attn_mask = file["data/attn_mask"][batch_idxs]
    return codes, graphs, info, vertices, attn_mask
    
    
def read_graph_info(fname: str, 
                    batch_idxs: Sequence[int]) -> Tuple[str, chex.Array, GraphInfo]:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:       
        graphs = file["data/graph"][batch_idxs]
        info = file["data/info"][batch_idxs]
        vertices = file["data/vertices"][batch_idxs]
        attn_mask = file["data/attn_mask"][batch_idxs]
        return graphs, info, vertices, attn_mask
    
    
def read_file_size(fname: str) -> int:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:
        header = file["header"]
        return header.attrs["num_samples"]

