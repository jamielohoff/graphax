from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex

from ..core import GraphInfo, make_graph_info


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
    return edges, new_info, vertex_mask[:num_intermediates-num_removed_vertices], attn_mask

