from typing import Sequence

import multiprocessing as mp

import jax
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
    
    def __init__(self, *args, debug: bool = False, num_cores: int = 8, **kwargs) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            key (PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.debug = debug
        self.num_cores = num_cores
        super().__init__(*args, **kwargs)
            
    def sample(self, 
                num_samples: int = 1, 
                key: PRNGKey = None,
                **kwargs) -> Sequence[tuple[str, Array]]:
        """
        Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """
        chunksize = num_samples//self.num_cores
        pool = mp.Pool(self.num_cores)
        samples = []
        keys = jrand.split(key, num_samples)
        it = [(key, self.max_info, kwargs) for key in keys]
        
        result = pool.starmap_async(sample_worker, it, chunksize=chunksize)
        pool.close()
        pool.join()
        
        samples = result.get()
        
        return samples
    
    
def sample_worker(key, max_info, kwargs):
    rkey, key = jrand.split(key, 2)
    code, jaxpr = make_random_code(rkey, max_info, **kwargs)
    edges = make_graph(jaxpr)

    edges = clean(edges)
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(key, edges, max_info)
    
    return code, edges
    
