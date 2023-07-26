from typing import Sequence
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from .utils import create, write
from ..interpreter import make_graph
from ..transforms import safe_preeliminations, compress_graph, embed, clean
from ..examples import make_random_code


def make_benchmark_dataset(key: PRNGKey, 
                            fname: str, 
                            size: int = 100,
                            max_info: Sequence[int] =[20, 105, 20]) -> None:
    samples = []
    create(fname, 100, max_info)
    
    # Do arbitrary operations
    for _ in tqdm(range(size//2)):
        ikey, vkey, okey, key = jrand.split(key, 4)
        num_i = jrand.randint(ikey, (), 2, 21)
        num_v = jrand.randint(vkey, (), 60, 105)
        num_o = jrand.randint(okey, (), 1, 21)
        info = [num_i, num_v, num_o]
        
        code, jaxpr = make_random_code(key, info)
        edges = make_graph(jaxpr)

        edges = clean(edges)
        edges = safe_preeliminations(edges)
        edges = compress_graph(edges)
        edges = embed(key, edges, max_info)
        
        samples.append((code, edges))
        
                
    # Do scalar only
    for _ in tqdm(range(size//2)):
        ikey, vkey, okey, key = jrand.split(key, 4)
        num_i = jrand.randint(ikey, (), 2, 21)
        num_v = jrand.randint(vkey, (), 90, 106)
        num_o = jrand.randint(okey, (), 1, 21)
        info = [num_i, num_v, num_o]
        
        code, jaxpr = make_random_code(key, 
                                        info, 
                                        primal_p=jnp.array([1., 0., 0.]), 
                                        prim_p=jnp.array([0.2, 0.8, 0., 0., 0.]))
        edges = make_graph(jaxpr)

        edges = clean(edges)
        edges = safe_preeliminations(edges)
        edges = compress_graph(edges)
        edges = embed(key, edges, max_info)
        
        samples.append((code, edges))
        
    write(fname, samples)   
    
    