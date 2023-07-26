import os
from typing import Sequence

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
    num_files: int
    samples_per_file: int
    storage_shape: Sequence[int]
    
    sampler_batchsize: int
    sampler: ComputationalGraphSampler
    
    def __init__(self, 
                sampler: ComputationalGraphSampler,
                path: str,
                fname_prefix: str = "comp_graph_examples", 
                num_samples: int = 16384,  
                storage_shape: Sequence[int] = [25, 130, 25]) -> None:
        self.path = path
        self.fname_prefix = fname_prefix
        self.num_samples = num_samples
        self.storage_shape = storage_shape
        self.sampler = sampler
        
    def generate(self, key: PRNGKey = None, **kwargs) -> None:
        handle = "_".join([str(s) for s in self.storage_shape])
        handle += "_" + str(self.num_samples)
        
        name = self.fname_prefix + "-" + handle + ".hdf5"
        fname = os.path.join(self.path, name)
        print("Saving under", fname)
        create(fname, num_samples=self.num_samples, max_info=self.storage_shape)
    
        subkey, key = jrand.split(key, 2)
        samples = self.sampler.sample(self.num_samples, key=subkey, **kwargs)
        print("Retrieved", len(samples), "samples")
        write(fname, samples)

