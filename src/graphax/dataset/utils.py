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

import numpy as np

from chex import Array

from ..core import make_empty_edges


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
        header_dset = file["data/graph_header"]
        graph_dset = file["data/graph"]
        
        code_dset[idx:idx+batchsize] = [sample[0] for sample in samples]
        header_dset[idx:idx+batchsize] = [sample[1] for sample in samples]
        graph_dset[idx:idx+batchsize] = [sample[2] for sample in samples]
    
        header.attrs["current_idx"] = idx + batchsize
        
        
def create(fname: str, num_samples: int, max_shape: Sequence[int] = (20, 105, 20)):
    assert os.path.isfile(fname) == False
    max_v = max_shape[1]
    
    with h5py.File(fname, "w") as file:
        header = file.create_group("header", (1,))
        header.attrs["num_samples"] = num_samples
        header.attrs["max_graph_shape"] = max_shape
        header.attrs["current_idx"] = 0
        
        data = file.create_group("data")
        str_dtype = h5py.string_dtype(encoding="utf-8")
        source_code = file.create_dataset("data/code", (num_samples,), dtype=str_dtype)
        
        header_dims = (5, max_v)
        graph_header = file.create_dataset("data/graph_header", (num_samples,)+header_dims, dtype="i4")
        
        graph_dtype = h5py.vlen_dtype(np.dtype('int32'))
        edges = file.create_dataset("data/graph", (num_samples,), dtype=graph_dtype)
        

def delete(fname: str):
    os.remove(fname)
    
    
def read(fname: str, batch_idxs: Sequence[int]) -> Tuple[str, Array]:
    """
    Codes variable needs to be decoded using .decode("utf-8")!
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname) as file:        
        codes = file["data/code"][batch_idxs]
        headers = file["data/graph_header"][batch_idxs]
        graphs = file["data/graph"][batch_idxs]
        return codes, headers, graphs
    
    
def read_graph(fname: str, batch_idxs: Sequence[int]) -> Tuple[Array]:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname) as file:     
        headers = file["data/graph_header"][batch_idxs]  
        graphs = file["data/graph"][batch_idxs]
        return headers, graphs
    
    
def read_file_size(fname: str) -> int:
    """
    TODO add documentation
    """
    assert os.path.isfile(fname) == True
    
    with h5py.File(fname, "a") as file:
        header = file["header"]
        return header.attrs["num_samples"]


def sparsify(edges: Array) -> Tuple[Array, Array]:
    """
    Function that takes in a 3d tensor that is the representation of a 
    computational graph and turns it into a sparsified version where we
    get a list of entries with corresponding values in the format
    (i, j, sparsity type, Jacobian shapes).
    This means it contains only existing edges.

    Args:
        edges (Array): Computational graph representation.
    """
    header = edges.at[:, 0, :].get()
    
    sparsity_map = edges.at[0, 1:, :].get()
    nonzeros = jnp.nonzero(sparsity_map)

    sparse_edges = []
    for i, j in zip(nonzeros[0], nonzeros[1]):
        edge = edges.at[:, i+1, j].get()
        idxs = jnp.array([i, j])
        edge = jnp.concatenate((idxs, edge))
        sparse_edges.append(edge)
    
    return header, jnp.concatenate(sparse_edges, axis=0)
    

def densify(header: Array, edges: Array, shape: Sequence[int] = [20, 105, 20]) -> Array:
    """
    Function that takes in the sparsified representation of a computational
    graph and turns it into a single dense 3d tensor again.
    Since data is loaded from hdf5 into numpy arrays, we use numpy to 
    reassemble the computational graph representation instead of jax.numpy.

    Args:
        header (Array): Computational graph representation.
        edges (Sequence[Array]): Computational graph representation containing
                                only nonzero entries, i.e. existing edges.
    """    
    dense_edges = np.array(make_empty_edges(shape))
    dense_edges[:, 0, :] = header

    for n in range(0, edges.shape[0], 7):
        edge = edges[n:n+7]
        i, j = edge[0:2]
        val = edge[2:]
        dense_edges[:, i+1, j] = val
    return dense_edges

