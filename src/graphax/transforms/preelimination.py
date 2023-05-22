from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex

from ..core import GraphInfo, vertex_eliminate_gpu, make_graph_info


def safe_preeliminations_gpu(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, int]:
    """
    Function that runs a safe-preelimination routing that eliminates all vertices
    with only one input and one output.
    WARNING: This changes the shape of the edges array and the number of intermediate variables!
    """
    num_intermediates = info.num_intermediates
    num_inputs = info.num_inputs
    
    def update_edges(carry, vertex):
        _edges = carry
        row_flag = jnp.sum(_edges[vertex+num_inputs, :]) == 1
        col_flag = jnp.sum(_edges[:, vertex]) == 1
        
        _edges, idx = lax.cond(jnp.logical_and(row_flag, col_flag),
                            lambda x: (vertex_eliminate_gpu(x, vertex+1, info)[0], vertex+1), 
                            lambda x: (x, 0), 
                            _edges)
        
        carry = _edges
        return carry, idx
    
    vertices = jnp.arange(0, num_intermediates)
    output, idxs = lax.scan(update_edges, edges, vertices)
    
    # Removes the zero rows and cols in the comp. graph repr.
    # TODO think about whether this is actually necessary or if we should
    # rather do this with attention masking?
    idxs = jnp.trim_zeros(idxs)
    for idx in idxs[::-1]:
        output = jnp.delete(output, idx-1+num_inputs, axis=0)
        output = jnp.delete(output, idx-1, axis=1)
    new_info = make_graph_info([info.num_inputs, output.shape[0]-info.num_inputs, info.num_outputs])
    return output, new_info, len(idxs)

