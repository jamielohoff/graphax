from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

# TODO needs adjustments
def connectivity_checker(edges: Array) -> Array:    
    """
    Function that checks if graph is fully connected
    """   
    num_i = edges.at[0, 0, 0].get()
    in_sum = jnp.sum(edges, axis=1)
    out_sum = jnp.sum(edges, axis=0)
    ins_connected = jnp.not_equal(in_sum, 0)[num_i:]
    outs_connected = jnp.not_equal(out_sum, 0)
    return jnp.logical_not(jnp.logical_xor(ins_connected, outs_connected))

# TODO needs adjustments
def clean(edges: Array) -> Array:
    """
    Removes all unconnected interior nodes
    """
    num_i, num_v, num_o = edges.at[0, 0, 0:3].get()
    
    conn = connectivity_checker(edges)
    is_clean = jnp.all(conn)
    while not is_clean:
        idxs = jnp.nonzero(jnp.logical_not(conn))
        
        def clean_edges_fn(_edges, idx):
            _edges = _edges.at[num_i + idx, :].set(jnp.zeros((1, num_v+num_o)))
            _edges = _edges.at[:, idx].set(jnp.zeros((num_i+num_v, 1)))
            return _edges, None
        edges, _ = lax.scan(clean_edges_fn, edges, idxs)
        conn = connectivity_checker(edges)
        is_clean = jnp.all(conn)
    return edges

