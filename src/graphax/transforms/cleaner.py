from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex

from ..core import GraphInfo


# function that checks if graph is fully connected
# scales linearly with number of intermediates
# quick implementation...
def connectivity_checker(edges: chex.Array, info: GraphInfo) -> chex.Array:       
	in_sum = jnp.sum(edges, axis=1)
	out_sum = jnp.sum(edges, axis=0)
	ins_connected = jnp.not_equal(in_sum, 0)[info.num_inputs:]
	outs_connected = jnp.not_equal(out_sum, 0)
	return jnp.logical_not(jnp.logical_xor(ins_connected, outs_connected))


# removes all unconnected interior nodes
def clean(edges: chex.Array, info: GraphInfo, vertex_mask: chex.Array, attn_mask: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
    conn = connectivity_checker(edges, info)
    is_clean = jnp.all(conn)
    while not is_clean:
        idxs = jnp.nonzero(jnp.logical_not(conn))
        
        def clean_edges_fn(_edges, idx):
            _edges = _edges.at[num_i + idx, :].set(jnp.zeros((1, num_v+num_o)))
            _edges = _edges.at[:, idx].set(jnp.zeros((num_i+num_v, 1)))
            return _edges, None
        edges, _ = lax.scan(clean_edges_fn, edges, idxs)
        conn = connectivity_checker(edges, info)
        is_clean = jnp.all(conn)
    return edges, info, vertex_mask, attn_mask

