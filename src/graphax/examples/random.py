from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex

@jax.jit
def construct_random_graph(key: chex.PRNGKey,
                            info: chex.Array, *, 
                            fraction: float = .35) -> Tuple[chex.Array, chex.Array]: 
    num_inputs, num_intermediates, num_outputs, _, _ = info
    in_key, var_key, out_key = jrand.split(key, 3)
    
    in_conns = jrand.uniform(in_key, (num_inputs, num_intermediates))
    in_conns = jnp.where(in_conns > fraction, 0, 1)
    
    var_conns = jrand.uniform(var_key, (num_intermediates, num_intermediates))
    var_conns = jnp.where(var_conns > fraction, 0, 1)
    var_conns = jnp.triu(var_conns, k=1)
    
    out_conns = jrand.uniform(out_key, (num_intermediates, num_outputs))
    out_conns = jnp.where(out_conns > fraction, 0, 1)
    
    edges = jnp.zeros((num_inputs+num_intermediates, num_intermediates+num_outputs), dtype=jnp.float32)    
    edges = edges.at[:num_inputs, :num_intermediates].set(in_conns)
    edges = edges.at[num_inputs:, :num_intermediates].set(var_conns)
    edges = edges.at[num_inputs:, num_intermediates:].set(out_conns)
        
    info[3] = jnp.sum(edges).astype(jnp.int32)
    return edges, info

