import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from ..core import get_shape


def connectivity_checker(edges: Array) -> Array:    
    """
    Function that checks if graph is fully connected
    """   
    num_i, num_vo = get_shape(edges)
    in_sum = jnp.sum(edges[0, 1:, :], axis=1)
    out_sum = jnp.sum(edges[0, 1:, :], axis=0)
    ins_connected = jnp.not_equal(in_sum, 0)[num_i:]
    outs_connected = jnp.not_equal(out_sum, 0)
    output_mask = edges.at[2, 0, :].get()
    is_connected = jnp.logical_xor(ins_connected, outs_connected)
    return jnp.logical_or(jnp.logical_not(is_connected), output_mask)


def clean(edges: Array) -> Array:
    """
    Removes all unconnected interior nodes
    """
    num_i, num_vo = get_shape(edges)
    row_shape = num_i+num_vo
        
    conn = connectivity_checker(edges)
    is_clean = jnp.all(conn)
    while not is_clean:
        idxs = jnp.nonzero(jnp.logical_not(conn))[0]
        def clean_edges_fn(_edges, idx):
            _edges = _edges.at[:, num_i + idx + 1, :].set(jnp.zeros((5, num_vo)))
            _edges = _edges.at[:, 1:, idx].set(jnp.zeros((5, row_shape)))
            return _edges, None
        edges, _ = lax.scan(clean_edges_fn, edges, idxs)
        conn = connectivity_checker(edges)
        is_clean = jnp.all(conn)

    return edges

