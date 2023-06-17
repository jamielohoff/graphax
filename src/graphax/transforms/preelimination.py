from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex

from ..core import GraphInfo, vertex_eliminate_gpu, make_graph_info


def remove_vertex_attn_mask(vertex: int, attn_mask: chex.Array) -> chex.Array:
    attn_mask = attn_mask.at[vertex-1, :].set(0.)
    attn_mask = attn_mask.at[:, vertex-1].set(0.)
    return attn_mask


def safe_preeliminations_gpu(edges: chex.Array, 
                            info: GraphInfo, 
                            vertex_mask: chex.Array, 
                            attn_mask: chex.Array) -> Tuple[chex.Array, GraphInfo, chex.Array, chex.Array]:
    """
    Function that runs a safe-preelimination routine that eliminates all vertices
    with only one input and one output.
    """
    num_intermediates = info.num_intermediates
    num_inputs = info.num_inputs
        
    def update_edges(carry, vertex):
        _edges, _attn_mask = carry
        
        # Remove vertices with Markowitz degree 1
        row = _edges[vertex+num_inputs-1, :]
        col = _edges[:, vertex-1]
        row_flag = jnp.equal(jnp.sum(jnp.where(row > 0, 1, 0)), 1)
        col_flag = jnp.equal(jnp.sum(jnp.where(col > 0, 1, 0)), 1)
        
        markowitz_degree_1 = jnp.logical_and(col_flag, row_flag) 
        
        # Remove dead branches from the computational graph
        row_flag = jnp.sum(_edges[vertex+num_inputs-1, :]) >= 1
        col_flag = jnp.sum(_edges[:, vertex-1]) == 0
        dead_branch = jnp.logical_and(row_flag, col_flag)
                
        __edges, __attn_mask, idx = lax.cond(jnp.logical_or(markowitz_degree_1, dead_branch),
                                    lambda x, a: (vertex_eliminate_gpu(vertex, x, info)[0], 
                                                remove_vertex_attn_mask(vertex, a), 
                                                vertex), 
                                    lambda x, a: (x, a, 0), 
                                    _edges, _attn_mask)        
        
        # Do not preeliminate output vertices
        is_output_vertex = jnp.any(vertex == vertex_mask)
        _edges, _attn_mask = lax.cond(is_output_vertex,
                            lambda: (_edges, _attn_mask),
                            lambda: (__edges, __attn_mask))
        
        carry = (_edges, _attn_mask)
        return carry, idx
    
    vertices = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(update_edges, (edges, attn_mask), vertices)
    edges, attn_mask = output
    return edges, info, vertex_mask, attn_mask


def compress_graph(edges: chex.Array, 
                   info: GraphInfo,
                   vertex_mask: chex.Array,
                   attn_mask: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    Function that removes all zero rows and cols from a comp. graph repr.
    WARNING: This changes the shape of the edges array and the number of intermediate variables!
    """
    num_inputs = info.num_inputs
    num_intermediates = info.num_intermediates
    num_outputs = info.num_outputs
            
    i, num_removed_vertices = 1, 0
    for _ in range(1, num_intermediates+1):            
        s1 = jnp.sum(edges.at[i+num_inputs-1, :].get()) == 0.
        s2 = jnp.sum(edges.at[:, i-1].get()) == 0.
        if s1 and s2:           
            add_mask = jnp.where(vertex_mask >= i, 1, 0)
            vertex_mask -= add_mask     
            edges = jnp.delete(edges, i+num_inputs-1, axis=0)
            edges = jnp.delete(edges, i-1, axis=1)
            attn_mask = jnp.delete(attn_mask, i-1, axis=0)
            attn_mask = jnp.delete(attn_mask, i-1, axis=1)
            num_removed_vertices += 1
        else:
            i += 1

    new_info = make_graph_info([num_inputs, num_intermediates-num_removed_vertices, num_outputs])
    return edges, new_info, lax.slice_in_dim(vertex_mask, 0, num_intermediates-num_removed_vertices), attn_mask

