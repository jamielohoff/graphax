from typing import Sequence
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrand

from chex import PRNGKey

from .utils import create, write
from .random_sampler import RandomSampler


def make_benchmark_dataset(key: PRNGKey, 
                            fname: str, 
                            size: int = 100,
                            max_shape: Sequence[int] =[20, 105, 20]) -> None:
    samples = []
    create(fname, size, max_shape)
    sampler = RandomSampler(min_num_intermediates=40, max_shape=max_shape)
    
    # Do scalars, vectors and matrices mixed
    result = sampler.sample(num_samples=size//2, key=key, sampling_shape=[20, 105, 20])
    samples.extend(result)
            
                
    # Do scalar only
    result = sampler.sample(num_samples=size//2, 
                            key=key, 
                            sampling_shape=[20, 105, 20],
                            primal_p=jnp.array([1., 0., 0.]), 
                            prim_p=jnp.array([.2, .8, 0., 0., 0.]))
    samples.extend(result)
        
    write(fname, samples)   
    
    