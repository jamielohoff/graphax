from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..core import GraphInfo

def construct_random(key: chex.PRNGKey,
                    info: GraphInfo, *, 
                    fraction: float = .35) -> Tuple[chex.Array, GraphInfo]: 
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    in_key, var_key, out_key = jrand.split(key, 3)
    
    in_conns = jrand.uniform(in_key, (num_i, num_v+num_o))
    in_conns = jnp.where(in_conns > fraction, 0, 1)
    
    var_conns = jrand.uniform(var_key, (num_v, num_v))
    var_conns = jnp.where(var_conns > fraction, 0, 1)
    var_conns = jnp.triu(var_conns, k=1)
    
    out_conns = jrand.uniform(out_key, (num_v, num_o))
    out_conns = jnp.where(out_conns > fraction, 0, 1)
    
    edges = jnp.zeros((num_i+num_v, num_v+num_o))    
    edges = edges.at[:num_i, :].set(in_conns)
    edges = edges.at[num_i:, :num_v].set(var_conns)
    edges = edges.at[num_i:, num_v:].set(out_conns)
    return edges, info

