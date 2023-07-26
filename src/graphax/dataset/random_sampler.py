from typing import Sequence

from tqdm import tqdm
import time

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
    
    def __init__(self, *args, debug: bool = False, **kwargs) -> None:
        """initializes a fixed repository of possible vertex games

        Args:
            num_games (int): _description_
            key (PRNGKey, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.debug = debug
        super().__init__(*args, **kwargs)
            
    def sample(self, 
                num_samples: int = 1, 
                key: PRNGKey = None,
                max_info: Sequence[int] = None,
                **kwargs) -> Sequence[tuple[str, Array]]:
        """
        Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """      
        samples = []
        max_info = self.max_info if max_info is None else max_info
        pbar = tqdm(total=num_samples)
        while len(samples) < num_samples:
            rkey, key = jrand.split(key, 2)
            print("Sampling...")
            st = time.time()
            try:
                code, jaxpr = make_random_code(rkey, max_info, **kwargs)
                print("Sampling time", time.time()-st)
                edges = make_graph(jaxpr)
                
                if self.debug:
                    print(code, edges)

                st = time.time()
                edges = clean(edges)
                print("Cleaning time", time.time()-st)
                
                st = time.time()
                edges = safe_preeliminations(edges)
                print("Preelimination time", time.time()-st)
                
                st = time.time()
                edges = compress_graph(edges)
                print("Compression time", time.time()-st)
                
                st = time.time()
                edges = embed(key, edges, self.max_info)
                print("Embedding time", time.time()-st)
            except Exception as e:
                print(e)
                continue
            
            if edges.at[0, 0, 1].get() >= self.min_num_intermediates:
                samples.append((code, edges))
                pbar.update(1)
        pbar.close()
        return samples
    
