from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from .core import GraphInfo
from .checker import connectivity_checker

# removes all unconnected interior nodes
def clean(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, GraphInfo]:
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
    conn = connectivity_checker(edges, info)
    is_clean = jnp.all(conn)
    while not is_clean:
        idxs = jnp.nonzero(jnp.logical_not(conn))
        
        for idx in idxs:
            edges = edges.at[num_i + idx, :].set(jnp.zeros((1, num_v+num_o)))
            edges = edges.at[:, idx].set(jnp.zeros((num_i+num_v, 1)))
        
        conn = connectivity_checker(edges, info)
        is_clean = jnp.all(conn)
        
    return edges, info

