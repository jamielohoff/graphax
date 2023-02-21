from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import add_edge


@jax.jit
def construct_simple() -> Tuple[chex.Array, chex. Array]:
    num_inputs = 2
    num_intermediates = 2
    num_outputs = 2
    info = jnp.array([num_inputs, num_intermediates, num_outputs, 0, 0])
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs))
    
    edges, info = add_edge(edges, (-1, 1), info)
    edges, info = add_edge(edges, (0, 1), info)

    edges, info = add_edge(edges, (1, 2), info)
    edges, info = add_edge(edges, (1, 3), info)

    edges, info = add_edge(edges, (2, 3), info)
    edges, info = add_edge(edges, (2, 4), info)
    return edges, info

