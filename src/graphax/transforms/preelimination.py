from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex

from ..core import GraphInfo, vertex_eliminate_gpu, make_graph_info


def safe_preeliminations_gpu(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, GraphInfo, int]:
    """
    Function that runs a safe-preelimination routing that eliminates all vertices
    with only one input and one output.
    WARNING: This changes the shape of the edges array and the number of intermediate variables!
    """
    num_intermediates = info.num_intermediates
    num_inputs = info.num_inputs
        
    def update_edges(carry, vertex):
        _edges = carry
        
        # remove vertices with markowitz degree 1, i.e. one input and one output
        row_flag = jnp.sum(_edges[vertex+num_inputs, :]) == 1
        col_flag = jnp.sum(_edges[:, vertex]) == 1
        markowitz_degree_1 = jnp.logical_and(row_flag, col_flag)
        
        # remove dead branches from the computational graph
        row_flag = jnp.sum(_edges[vertex+num_inputs, :]) >= 1
        col_flag = jnp.sum(_edges[:, vertex]) == 0
        dead_branch = jnp.logical_and(row_flag, col_flag)
        
        _edges, idx = lax.cond(jnp.logical_or(markowitz_degree_1, dead_branch),
                            lambda x: (vertex_eliminate_gpu(x, vertex+1, info)[0], vertex+1), 
                            lambda x: (x, 0), 
                            _edges)
        
        carry = _edges
        return carry, idx
    
    vertices = jnp.arange(1, num_intermediates)
    output, _ = lax.scan(update_edges, edges, vertices)
    
    return output, info


def compress_graph(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, GraphInfo]:
    """
    Function that removes all zero rows and cols from a comp. graph repr.
    """
    num_inputs = info.num_inputs
    num_intermediates = info.num_intermediates
    num_outputs = info.num_outputs

    i, num_removed_vertices = 0, 0
    for _ in range(num_intermediates):            
        s1 = jnp.sum(edges.at[i+num_inputs, :].get()) == 0.
        s2 = jnp.sum(edges.at[:, i].get()) == 0.
        if s1 and s2:
            edges = jnp.delete(edges, i+num_inputs, axis=0)
            edges = jnp.delete(edges, i, axis=1)
            num_removed_vertices += 1
        else:
            i += 1
    new_info = make_graph_info([num_inputs, num_intermediates-num_removed_vertices, num_outputs])
    return edges, new_info

