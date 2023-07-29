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
                sampling_shape: Sequence[int] = [20, 105, 20],
                **kwargs) -> Sequence[tuple[str, Array]]:
        """
        Samples from the repository of possible games

        Args:
            x (_type_): _description_

        Returns:
            Any: _description_
        """      
        samples = []
        pbar = tqdm(total=num_samples)
        while len(samples) < num_samples:
            rkey, key = jrand.split(key, 2)
            print("Sampling...")
            st = time.time()
            try:
                ikey, vkey, okey, key = jrand.split(key, 4)
                num_i = jrand.randint(ikey, (), 2, sampling_shape[0]+1)
                num_v = sampling_shape[1]
                num_o = jrand.randint(okey, (), 1, sampling_shape[2])
                shape = [num_i, num_v, num_o]
                code, jaxpr = make_random_code(rkey, shape, **kwargs)
                                
                # print("Sampling time", time.time()-st)
                edges = make_graph(jaxpr)
                del jaxpr
                
                if self.debug:
                    print(code, edges)

                st = time.time()
                edges = clean(edges)
                # print("Cleaning time", time.time()-st)
                
                st = time.time()
                edges = safe_preeliminations(edges)
                # print("Preelimination time", time.time()-st)
                
                large_enough = edges.at[0, 0, 1].get() >= self.min_num_intermediates
                if large_enough:
                    st = time.time()
                    edges = compress_graph(edges)
                    # print("Compression time", time.time()-st)
                    
                    st = time.time()
                    edges = embed(key, edges, self.storage_shape)
                    # print("Embedding time", time.time()-st)
                    samples.append((code, edges))
                    print(f"{len(samples)}/{num_samples} samples")
                    pbar.update(1)
                else: 
                    print("Sample of shape", edges.at[0, 0, 0:3].get().tolist(), "rejected!")
                    continue
            except Exception as e:
                print(e)
                del code
                del edges
                continue

            del code
            del edges
        pbar.close()
        return samples
    
