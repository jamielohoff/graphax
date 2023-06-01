from typing import Sequence

import multiprocessing as mp

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from .sampler import ComputationalGraphSampler
from ..core import GraphInfo
from ..examples import make_random
from ..transforms import safe_preeliminations_gpu, compress_graph, embed, clean


def _sample(input):
    max_info, key, kwargs = input
    fkey, rkey, key = jrand.split(key, 3)
    fraction = jrand.uniform(fkey, (), **kwargs)
    edges, info = make_random(rkey, max_info, fraction=fraction)
    edges, info = clean(edges, info)
    edges, info = safe_preeliminations_gpu(edges, info)
    edges, info = compress_graph(edges, info)
    edges, _, vertices, attn_mask = embed(edges, info, max_info)
    return "", edges, info, vertices, attn_mask


class RandomSampler(ComputationalGraphSampler):
    """
    TODO add documentation
    """
    num_cores: int
    
    def __init__(self, num_cores: int, *args, **kwargs) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            info (chex.Array): _description_
            key (chex.PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        super().__init__(*args, **kwargs)
        self.num_cores = num_cores
            
    def sample(self, 
                num_samples: int = 1, 
                key: chex.PRNGKey = None,
                **kwargs) -> Sequence[tuple[str, chex.Array, GraphInfo]]:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        samples = []
        
        keys = jrand.split(key, num_samples)
        input = [(self.max_info, key, kwargs) for key in keys]
        with mp.Pool(self.num_cores) as pool:
            samples = pool.map(_sample, input)
            
        samples = [sample for sample in samples if sample[2].num_intermediates > self.min_num_intermediates]
        
        # while len(samples) < num_samples:
        #     fkey, rkey, key = jrand.split(key, 3)
        #     fraction = jrand.uniform(fkey, (), **kwargs)
        #     edges, info = make_random(rkey, self.max_info, fraction=fraction)
        #     edges, info = clean(edges, info)
        #     edges, info = safe_preeliminations_gpu(edges, info)
        #     edges, info = compress_graph(edges, info)
        #     num_intermediates = info.num_intermediates
        #     edges, _, vertices, attn_mask = embed(edges, info, self.max_info)
        #     print(info)
        #     num_intermediates = info.num_intermediates
        #     if num_intermediates > self.min_num_intermediates:
        #         samples.append(("", edges, info, vertices, attn_mask))
        return samples
    
