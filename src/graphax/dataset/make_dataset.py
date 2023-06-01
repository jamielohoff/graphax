import os
from tqdm import tqdm

import chex

from .sampler import ComputationalGraphSampler
from .utils import create, write
from ..core import GraphInfo, make_graph_info

class Graph2File:
    """
    Class to create a large dataset of computational graphs.
    """
    path: str
    fname_prefix: str
    num_samples: int
    num_files: int
    samples_per_file: int
    max_info: GraphInfo
    
    sampler_batchsize: int
    sampler: ComputationalGraphSampler
    
    def __init__(self, 
                sampler: ComputationalGraphSampler,
                path: str,
                sampler_batchsize: int = 20,
                fname_prefix: str = "comp_graph_examples", 
                num_samples: int = 200,  
                samples_per_file: int = 100,
                max_info: GraphInfo = make_graph_info([10, 30, 5])) -> None:
        self.path = path
        self.fname_prefix = fname_prefix
        self.num_samples = num_samples
        self.samples_per_file = samples_per_file
        self.num_files = num_samples // samples_per_file
        self.current_num_files = 0
        self.max_info = max_info
        
        self.sampler_batchsize = sampler_batchsize
        self.sampler = sampler
        
    def generate(self, key: chex.PRNGKey = None, **kwargs) -> None:
        pbar = tqdm(range(self.num_files))
        for _ in pbar:
            fname = self.new_file()
            num_current_samples = 0
            while num_current_samples < self.samples_per_file:
                try:
                    samples = self.sampler.sample(self.sampler_batchsize, key=key, **kwargs)
                except Exception as e:
                    print(e)
                    continue
                print("Retrieved", len(samples), "samples")
                num_current_samples += len(samples)
                write(fname, samples)
                
    def new_file(self) -> str:
        name = self.fname_prefix + "-" + str(self.current_num_files) + ".hdf5"
        fname = os.path.join(self.path, name)
        create(fname, num_samples=self.samples_per_file, max_info=self.max_info)
        self.current_num_files += 1
        return fname

