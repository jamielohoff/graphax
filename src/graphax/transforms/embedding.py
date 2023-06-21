from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..core import GraphInfo


def embed(edges: chex.Array,
        info: GraphInfo,
        new_info: GraphInfo,
        vertex_mask: chex.Array,
        attn_mask: chex.Array) -> Tuple[chex.Array, GraphInfo, chex.Array, chex.Array]:
    """
    Creates a deterministic embedding that is compatible with a sequence transformer
    WARNING: This is deprecated as it is not a good idea to use this since it 
    creates an unwanted inductive bias in the system.
    """
    num_i = info.num_inputs
    num_v = info.num_intermediates
    
    if info == new_info:
        attn_mask = jnp.ones((num_v, num_v))
        return edges, info, vertex_mask, attn_mask
    
    new_num_i = new_info.num_inputs
    new_num_v = new_info.num_intermediates
        
    i_diff = new_num_i - num_i
    v_diff = new_num_v - num_v

    le, re = jnp.vsplit(edges, (num_i,))
    edges = jnp.concatenate([le, jnp.zeros((i_diff, num_v)), re], axis=0)
    
    te, be = jnp.hsplit(edges, (num_v,))
    edges = jnp.concatenate([te, jnp.zeros((new_num_i+num_v, v_diff)), be], axis=1)
        
    edges = jnp.append(edges, jnp.zeros((v_diff, new_num_v)), axis=0)
    
    zeros = jnp.zeros((num_v, v_diff))
    attn_mask = jnp.concatenate((attn_mask, zeros), axis=1)
    zeros = jnp.zeros((v_diff, new_num_v))
    attn_mask = jnp.concatenate((attn_mask, zeros), axis=0)
    
    vertices = 1. + jnp.nonzero(1. - attn_mask.at[0, :].get())[0]
    vertex_mask = jnp.append(vertices, vertex_mask)
    
    return edges, new_info, vertex_mask, attn_mask
    

def random_embed(key: chex.PRNGKey, 
                edges: chex.Array,
                info: GraphInfo,
                new_info: GraphInfo,
                vertex_mask: chex.Array,
                attn_mask: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    Embeds a smaller graph into a larger graph frame based on random inserts
    """
    ikey, vkey, okey = jrand.split(key, 3)
    
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
    new_num_i = new_info.num_inputs
    new_num_v = new_info.num_intermediates
        
    # properly adjust the vertex mask
    
    i_diff = new_num_i - num_i
    v_diff = new_num_v - num_v
    
    i_split_idxs = jrand.randint(ikey, (i_diff,), 0, num_i)
    v_split_idxs = jrand.randint(vkey, (v_diff,), new_num_i, new_num_i+num_v)
    
    for i in i_split_idxs:
        le, re = jnp.vsplit(edges, (i,))
        edges = jnp.concatenate([le, jnp.zeros((1, num_v+num_o)), re], axis=0)
        
    for e, v in enumerate(v_split_idxs):
        le, re = jnp.vsplit(edges, (v,))
        edges = jnp.concatenate([le, jnp.zeros((1, num_v+num_o+e)), re], axis=0)
        te, be = jnp.hsplit(edges, (v-new_num_i,))
        edges = jnp.concatenate([te, jnp.zeros((new_num_i+num_v+e+1, 1)), be], axis=1)
        
    # select some of the new vertices as new output vertices
    
    return edges, new_info

