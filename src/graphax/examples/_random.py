from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..core import GraphInfo, make_empty_edges, add_edge


def make_random(key: chex.PRNGKey,
                info: GraphInfo, 
                *, 
                fraction: float = .35) -> Tuple[chex.Array, GraphInfo]: 
    in_key, var_key, out_key = jrand.split(key, 3)
    
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
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


def make_connected_random(key: chex.PRNGKey,
                        info: GraphInfo, *, 
                        max_connections: int = 4,
                        p: chex.Array = None) -> Tuple[chex.Array, GraphInfo]:
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    edges = make_empty_edges(info)
    
    size_choices = jnp.arange(1, max_connections, 1)
    sizes = jrand.choice(key, size_choices, (num_i+num_v,), p=p)
    
    for i, size in zip(range(-num_i+1, num_v+1), sizes):
        subkey, key = jrand.split(key, 2)
        
        lb = i+1 if i > 0 else 1
        choices = jnp.arange(lb, num_v + num_o + 1, 1)
        # p_dist = 
        js = jrand.choice(subkey, choices, (size,), replace=False, p=None)
        for j in js:
            edges = add_edge(edges, (i, j), info)
        
    return edges, info
    
