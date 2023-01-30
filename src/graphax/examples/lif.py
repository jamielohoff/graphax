from typing import Tuple

import jax
import jax.numpy as jnp

from ..graph import GraphState, add_edge


@jax.jit
def construct_LIF() -> Tuple[GraphState, int]:
    num_inputs = 6
    num_intermediates = 9
    num_outputs = 3
    info = jnp.array([num_inputs, num_intermediates, num_outputs, 0, 0])
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs), dtype=jnp.float32)
    state = jnp.zeros((num_intermediates,))
    gs = GraphState(info, edges, state)
    
    gs = add_edge(gs, (0,6), 1., info)
    
    gs = add_edge(gs, (1,6), 1., info)
    gs = add_edge(gs, (1,7), 1., info)
    
    gs = add_edge(gs, (2,8), 1., info)
    gs = add_edge(gs, (2,10), 1., info)
    
    gs = add_edge(gs, (3,8), 1., info)
    gs = add_edge(gs, (3,9), 1., info)
    
    gs = add_edge(gs, (4,11), 1., info)
    
    gs = add_edge(gs, (5,14), 1., info)
    
    gs = add_edge(gs, (6,12), 1., info)
    
    gs = add_edge(gs, (7,10), 1., info)
    
    gs = add_edge(gs, (8,13), 1., info)
    
    gs = add_edge(gs, (9,11), 1., info)
    
    gs = add_edge(gs, (10,12), 1., info)
    
    gs = add_edge(gs, (11,13), 1., info)
    
    gs = add_edge(gs, (12,14), 1., info)
    gs = add_edge(gs, (12,16), 1., info)
    
    gs = add_edge(gs, (13,17), 1., info)
    
    gs = add_edge(gs, (14,15), 1., info)
    gs = add_edge(gs, (14,16), 1., info) # gated reset

    return gs, 20

