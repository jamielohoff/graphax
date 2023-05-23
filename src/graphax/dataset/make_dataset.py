import os
import random
import time
from time import sleep
from typing import Sequence, Tuple
from tqdm import tqdm

from .llm_sampler import ComputationalGraphSampler
from .utils import create, write

class Graph2File:
    """
    Class to create a large dataset of computational graphs.
    """
    path: str
    fname_prefix: str
    prompt_list: Sequence[Tuple[str, str]]
    num_samples: int
    num_files: int
    samples_per_file: int
    max_graph_shape: Tuple[int, int, int]
    
    sampler_batchsize: int
    sampler: ComputationalGraphSampler
    
    def __init__(self, 
                api_key: str,
                path: str,
                prompt_list: Sequence[Tuple[str, str]],
                sampler_batchsize: int = 10,
                fname_prefix: str = "comp_graph_examples", 
                num_samples: int = 200,  
                samples_per_file: int = 20,
                max_graph_shape: Tuple[int, int, int] = (10, 15, 5)) -> None:
        self.path = path
        self.prompt_list = prompt_list
        self.fname_prefix = fname_prefix
        self.num_samples = num_samples
        self.samples_per_file = samples_per_file
        self.num_files = num_samples // samples_per_file
        self.max_graph_shape = max_graph_shape
        self.current_num_files = 0
        
        self.sampler_batchsize = sampler_batchsize
        default_message = prompt_list[0][0]
        default_make_jaxpr = prompt_list[0][1]
        self.sampler = ComputationalGraphSampler(api_key, 
                                                default_message, 
                                                default_make_jaxpr,
                                                max_graph_shape)
        
    def generate(self) -> None:
        pbar = tqdm(range(self.num_files))
        for f in pbar:
            fname = self.new_file()
            nbar = tqdm(range(self.samples_per_file//self.sampler_batchsize))
            for s in nbar:
                idx = random.randint(0, len(self.prompt_list)-1)
                samples = []
                try:
                    samples = self.sampler.sample(self.sampler_batchsize, 
                                                message=self.prompt_list[idx][0],
                                                make_jaxpr=self.prompt_list[idx][1])
                except SyntaxError:
                    s -= 1
                write(fname, samples)

    def new_file(self) -> str:
        name = self.fname_prefix + "-" + str(self.current_num_files) + ".hdf5"
        fname = os.path.join(self.path, name)
        create(fname, num_samples=self.samples_per_file, max_graph_shape=self.max_graph_shape)
        self.current_num_files += 1
        return fname

