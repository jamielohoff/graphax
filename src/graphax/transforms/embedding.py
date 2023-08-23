from typing import Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from ..core import get_shape
    

def embed(key: PRNGKey, edges: Array, new_size: Sequence[int]) -> Array:
    """
    Embeds a smaller graph into a larger graph frame based on random inserts
    NOTE: Changes size of the tensor!
    """
    
    ikey, vkey = jrand.split(key, 2)
    num_i, num_vo = get_shape(edges)
    size = edges.at[0, 0, 0:3].get()
    new_num_i, new_num_vo, new_num_o = new_size

    i_diff = new_num_i - num_i
    vo_diff = new_num_vo - num_vo

    i_split_idxs = jrand.randint(ikey, (i_diff,), 1, num_i+1)
    v_split_idxs = jrand.randint(vkey, (vo_diff,), new_num_i+1, new_num_i+num_vo+1)
    
    for i in i_split_idxs:
        le, re = jnp.split(edges, (i,), axis=1)
        edges = jnp.concatenate([le, jnp.zeros((5, 1, num_vo), dtype=jnp.int32), re], axis=1)
        
    for e, v in enumerate(v_split_idxs):
        le, re = jnp.split(edges, (v,), axis=1)
        edges = jnp.concatenate([le, jnp.zeros((5, 1, num_vo+e), dtype=jnp.int32), re], axis=1)
        te, be = jnp.split(edges, (v-new_num_i-1,), axis=2)
        edges = jnp.concatenate([te, jnp.zeros((5, new_num_i+1+num_vo+e+1, 1), dtype=jnp.int32), be], axis=2)
        edges = edges.at[1, 0, v-new_num_i-1].set(1)
        
    # Update edge state size to new size
    edges = edges.at[0, 0, :].set(0)
    edges = edges.at[0, 0, 0:3].set(jnp.array(size))
    return edges

