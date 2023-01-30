from typing import Tuple

import jax
import jax.numpy as jnp

from ..graph import GraphState, add_edge


@jax.jit
def construct_simple() -> Tuple[GraphState, int]:
    num_inputs = 2
    num_intermediates = 2
    num_outputs = 2
    info = jnp.array([num_inputs, num_intermediates, num_outputs, 0, 0])
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs), dtype=jnp.float32)
    state = jnp.zeros(num_intermediates)
    gs = GraphState(info, edges, state)
    
    gs = add_edge(gs, (-1,1), 1., info)
    gs = add_edge(gs, (0,1), 1., info)

    gs = add_edge(gs, (1,2), 1., info)
    gs = add_edge(gs, (1,3), 1., info)

    gs = add_edge(gs, (2,3), 1., info)
    gs = add_edge(gs, (2,4), 1., info)
    return gs

