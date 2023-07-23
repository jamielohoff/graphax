from typing import Sequence

import multiprocessing as mp

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from .sampler import ComputationalGraphSampler
from ..examples import make_random_code
from ..transforms import safe_preeliminations, compress_graph, embed, clean
from ..interpreter.from_jaxpr import make_graph


class RandomSampler(ComputationalGraphSampler):
    """
    TODO add documentation
    TODO use multiprocessing
    """
    
    def __init__(self, *args, **kwargs) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            info (Array): _description_
            key (PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        super().__init__(*args, **kwargs)
            
    def sample(self, 
                num_samples: int = 1, 
                key: PRNGKey = None,
                **kwargs) -> Sequence[tuple[str, Array]]:
        """Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        samples = []
        
        while len(samples) < num_samples:
            rkey, key = jrand.split(key, 2)
            code, jaxpr = make_random_code(rkey, self.max_info, **kwargs)
            edges = make_graph(jaxpr)
            print(code, edges)
            
            # TODO check these
            edges = clean(edges)
            edges = safe_preeliminations(edges)
            edges = compress_graph(edges)
            
            edges = embed(edges, self.max_info)
            if edges.at[0, 0, 1].get() >= self.min_num_intermediates:
                samples.append((code, edges))
        return samples
    
