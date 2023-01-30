from typing import Tuple

import jax
import jax.numpy as jnp

from ..graph import GraphState, add_edge

@jax.jit
def construct_Helmholtz() -> Tuple[GraphState, int]:
    num_inputs = 4
    num_intermediates = 11
    num_outputs = 4
    info = jnp.array([num_inputs, num_intermediates, num_outputs, 0, 0])
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs), dtype=jnp.float32)
    state = jnp.zeros((num_intermediates,))
    gs = GraphState(info, edges, state)
    
    gs = add_edge(gs, (-3,1), 1., info)
    gs = add_edge(gs, (-3,4), 1., info)
    gs = add_edge(gs, (-3,12), 1., info)
    
    gs = add_edge(gs, (-2,1), 1., info)
    gs = add_edge(gs, (-2,5), 1., info)
    gs = add_edge(gs, (-2,13), 1., info)
    
    gs = add_edge(gs, (-1,1), 1., info)
    gs = add_edge(gs, (-1,6), 1., info)
    gs = add_edge(gs, (-1,14), 1., info)
    
    gs = add_edge(gs, (0,1), 1., info)
    gs = add_edge(gs, (0,7), 1., info)
    gs = add_edge(gs, (0,15), 1., info)
    
    gs = add_edge(gs, (1,2), 1., info)
    
    gs = add_edge(gs, (2,3), 1., info)
    
    gs = add_edge(gs, (3,4), 1., info)
    gs = add_edge(gs, (3,5), 1., info)
    gs = add_edge(gs, (3,6), 1., info)
    gs = add_edge(gs, (3,7), 1., info)
    
    gs = add_edge(gs, (4,8), 1., info)
    gs = add_edge(gs, (5,9), 1., info)
    gs = add_edge(gs, (6,10), 1., info)
    gs = add_edge(gs, (7,11), 1., info)
    
    gs = add_edge(gs, (8,12), 1., info)
    gs = add_edge(gs, (9,13), 1., info)
    gs = add_edge(gs, (10,14), 1., info)
    gs = add_edge(gs, (11,15), 1., info)
    return gs

