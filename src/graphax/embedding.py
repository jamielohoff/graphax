from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex

from .core import GraphInfo

# embeds a smaller graph into a larger graph frame
def embed(key: chex.PRNGKey, 
        edges: chex.Array,
        info: GraphInfo,
        new_info: GraphInfo) -> Tuple[chex.Array, GraphInfo]:
    ikey, vkey, okey = jrand.split(key, 3)
    
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
    new_num_i = new_info.num_inputs
    new_num_v = new_info.num_intermediates
    new_num_o = new_info.num_outputs
    
    i_diff = new_num_i - num_i
    v_diff = new_num_v - num_v
    o_diff = new_num_o - num_o
    
    i_split_idxs = jrand.randint(ikey, (i_diff,), 0, num_i)
    v_split_idxs = jrand.randint(vkey, (v_diff,), new_num_i, new_num_i+num_v)
    o_split_idxs = jrand.randint(okey, (o_diff,), new_num_v, new_num_v+num_o)
    
    for i in i_split_idxs:
        le, re = jnp.vsplit(edges, (i,))
        edges = jnp.concatenate([le, jnp.zeros((1, num_v+num_o)), re], axis=0)
        
    for e, v in enumerate(v_split_idxs):
        le, re = jnp.vsplit(edges, (v,))
        edges = jnp.concatenate([le, jnp.zeros((1, num_v+num_o+e)), re], axis=0)
        te, be = jnp.hsplit(edges, (v-new_num_i,))
        edges = jnp.concatenate([te, jnp.zeros((new_num_i+num_v+e+1, 1)), be], axis=1)
        
    for o in o_split_idxs:
        te, be = jnp.hsplit(edges, (o,))
        edges = jnp.concatenate([te, jnp.zeros((new_num_i+new_num_v, 1)), be], axis=1)
    
    return edges, new_info
    
