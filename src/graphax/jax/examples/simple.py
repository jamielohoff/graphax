from typing import Tuple

import jax
import jax.numpy as jnp

from ..graph import GraphState, add_edge


@jax.jit
def construct_simple() -> Tuple[GraphState, int]:
    ninputs = 2
    nintermediates = 2
    noutputs = 2
    info = jnp.array([ninputs, nintermediates, noutputs, 0, 0])
    edges = jnp.zeros((ninputs+nintermediates, nintermediates+noutputs), dtype=jnp.float32)
    state = jnp.zeros((nintermediates,))
    gs = GraphState(info, edges, state)
    
    gs = add_edge(gs, (-1,1), 1., info)
    gs = add_edge(gs, (0,1), 1., info)

    gs = add_edge(gs, (1,2), 1., info)
    gs = add_edge(gs, (1,3), 1., info)

    gs = add_edge(gs, (2,3), 1., info)
    gs = add_edge(gs, (2,4), 1., info)
    return gs

