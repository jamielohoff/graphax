import os
import random
from typing import Sequence, Tuple
from tqdm import tqdm

from ..core import GraphInfo, make_graph_info
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
    max_info: GraphInfo
    
    sampler_batchsize: int
    sampler: ComputationalGraphSampler
    
    def __init__(self, 
                api_key: str,
                path: str,
                prompt_list: Sequence[Tuple[str, str]],
                sampler_batchsize: int = 20,
                fname_prefix: str = "comp_graph_examples", 
                num_samples: int = 200,  
                samples_per_file: int = 100,
                max_info: GraphInfo = make_graph_info((10, 30, 5))) -> None:
        self.path = path
        self.prompt_list = prompt_list
        self.fname_prefix = fname_prefix
        self.num_samples = num_samples
        self.samples_per_file = samples_per_file
        self.num_files = num_samples // samples_per_file
        self.max_info = max_info
        self.current_num_files = 0
        
        self.sampler_batchsize = sampler_batchsize
        default_message = prompt_list[0][0]
        default_make_jaxpr = prompt_list[0][1]
        self.sampler = ComputationalGraphSampler(api_key, 
                                                default_message, 
                                                default_make_jaxpr,
                                                max_info)
        
    def generate(self) -> None:
        pbar = tqdm(range(self.num_files))
        num_current_samples = 0
        for _ in pbar:
            fname = self.new_file()
            while num_current_samples < self.samples_per_file:
                idx = random.randint(0, len(self.prompt_list)-1)
                samples = self.sampler.sample(self.sampler_batchsize, 
                                            message=self.prompt_list[idx][0],
                                            make_jaxpr=self.prompt_list[idx][1])
                print("Retrieved", len(samples), "samples")
                num_current_samples += len(samples)
                write(fname, samples)

    def new_file(self) -> str:
        name = self.fname_prefix + "-" + str(self.current_num_files) + ".hdf5"
        fname = os.path.join(self.path, name)
        create(fname, num_samples=self.samples_per_file, max_info=self.max_info)
        self.current_num_files += 1
        return fname

