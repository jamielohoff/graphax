from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import add_edge

@jax.jit
def construct_Helmholtz() -> Tuple[chex.Array, chex.Array]:
    num_inputs = 4
    num_intermediates = 11
    num_outputs = 4
    info = jnp.array([num_inputs, num_intermediates, num_outputs, 0, 0])
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs))
    
    edges, info = add_edge(edges, (-3, 1), info)
    edges, info = add_edge(edges, (-3, 4), info)
    edges, info = add_edge(edges, (-3, 12), info)
    
    edges, info = add_edge(edges, (-2, 1), info)
    edges, info = add_edge(edges, (-2, 5), info)
    edges, info = add_edge(edges, (-2, 13), info)
    
    edges, info = add_edge(edges, (-1, 1), info)
    edges, info = add_edge(edges, (-1, 6), info)
    edges, info = add_edge(edges, (-1, 14), info)
    
    edges, info = add_edge(edges, (0, 1), info)
    edges, info = add_edge(edges, (0, 7), info)
    edges, info = add_edge(edges, (0, 15), info)
    
    edges, info = add_edge(edges, (1, 2), info)
    
    edges, info = add_edge(edges, (2, 3), info)
    
    edges, info = add_edge(edges, (3, 4), info)
    edges, info = add_edge(edges, (3, 5), info)
    edges, info = add_edge(edges, (3, 6), info)
    edges, info = add_edge(edges, (3, 7), info)
    
    edges, info = add_edge(edges, (4, 8), info)
    edges, info = add_edge(edges, (5, 9), info)
    edges, info = add_edge(edges, (6, 10), info)
    edges, info = add_edge(edges, (7, 11), info)
    
    edges, info = add_edge(edges, (8, 12), info)
    edges, info = add_edge(edges, (9, 13), info)
    edges, info = add_edge(edges, (10, 14), info)
    edges, info = add_edge(edges, (11, 15), info)
    return edges, info

