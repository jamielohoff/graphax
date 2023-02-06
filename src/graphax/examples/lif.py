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
    state = jnp.zeros(num_intermediates)
    gs = GraphState(info, edges, state)
    
    gs = add_edge(gs, (-5,1), 1., info)
    
    gs = add_edge(gs, (-4,1), 1., info)
    gs = add_edge(gs, (-4,2), 1., info)
    
    gs = add_edge(gs, (-3,3), 1., info)
    gs = add_edge(gs, (-3,5), 1., info)
    
    gs = add_edge(gs, (-2,3), 1., info)
    gs = add_edge(gs, (-2,4), 1., info)
    
    gs = add_edge(gs, (-1,6), 1., info)
    
    gs = add_edge(gs, (0,9), 1., info)
    
    gs = add_edge(gs, (1,7), 1., info)
    
    gs = add_edge(gs, (2,5), 1., info)
    
    gs = add_edge(gs, (3,8), 1., info)
    
    gs = add_edge(gs, (4,6), 1., info)
    
    gs = add_edge(gs, (5,7), 1., info)
    
    gs = add_edge(gs, (6,8), 1., info)
    
    gs = add_edge(gs, (7,9), 1., info)
    gs = add_edge(gs, (7,11), 1., info)
    
    gs = add_edge(gs, (8,12), 1., info)
    
    gs = add_edge(gs, (9,10), 1., info)
    gs = add_edge(gs, (9,11), 1., info) # gated reset

    return gs
