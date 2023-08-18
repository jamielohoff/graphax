import os
from typing import Sequence

import gc

import jax
import jax.random as jrand

from chex import PRNGKey

from .sampler import ComputationalGraphSampler
from .utils import create, write


class Graph2File:
    """
    Class to create a large dataset of computational graphs.
    """
    path: str
    fname_prefix: str
    num_samples: int
    batchsize: int
    storage_shape: Sequence[int]
    sampler: ComputationalGraphSampler
    
    def __init__(self, 
                sampler: ComputationalGraphSampler,
                path: str,
                fname_prefix: str = "comp_graph_examples", 
                num_samples: int = 16384,  
                batchsize: int = 1,
                storage_shape: Sequence[int] = [20, 105, 20]) -> None:
        self.path = path
        self.fname_prefix = fname_prefix
        self.num_samples = num_samples
        self.storage_shape = storage_shape
        self.sampler = sampler
        self.batchsize = batchsize
        
    def generate(self, key: PRNGKey = None, **kwargs) -> None:
        ri = int(jrand.randint(key, (), 0, 1e6))
        handle = "_".join([str(s) for s in self.storage_shape])
        handle += f"_{self.num_samples}"
        handle += f"_{ri}"
        
        name = self.fname_prefix + "-" + handle + ".hdf5"
        fname = os.path.join(self.path, name)
        print("Saving under", fname)
        create(fname, num_samples=self.num_samples, max_info=self.storage_shape)
    
        subkey, key = jrand.split(key, 2)
        
        num_samples = 0
        while num_samples < self.num_samples:
            samples = self.sampler.sample(self.batchsize, key=subkey, **kwargs)
            print("Writing", len(samples), "samples to file...")
            num_samples += len(samples)
            write(fname, samples)
            
            del samples
            gc.collect()

