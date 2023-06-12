from typing import Tuple

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from .core import GraphInfo, vertex_eliminate_gpu


def random_solver(edges: chex.Array,
                  info: GraphInfo,
                  num_iterations: int = 500,
                  key: chex.PRNGKey = None) -> Tuple[chex.Array, float]:
    """Tries to find the optimal elimination order using a simple random search.

    Args:
        edges (chex.Array): _description_
        info (GraphInfo): _description_
        num_iterations (int, optional): _description_. Defaults to 500.
        key (chex.PRNGKey, optional): _description_. Defaults to None.

    Returns:
        Tuple[chex.Array, float]: _description_
    """
    num_intermediates = info.num_intermediates
    best_order = jnp.zeros(num_intermediates, dtype=jnp.int32)
    
    def inner_loop_fn(carry, vertex):
        edges, nops = carry
        vertex = vertex.astype(jnp.int32)
        edges, ops = vertex_eliminate_gpu(edges, vertex, info)
        nops += ops
        carry = (edges, nops)
        return carry, None
    
    def outer_loop_fn(carry, _):
        best_nops, best_order, key = carry
        subkey, key = jrand.split(key)
        elimination_order = jrand.choice(subkey, jnp.arange(0, num_intermediates), (num_intermediates,), replace=False)
        (_, nops), _ = lax.scan(inner_loop_fn, (edges, 0.), elimination_order, length=num_intermediates)
        best_nops, best_order = lax.cond(nops <= best_nops, 
                                        lambda: (nops, elimination_order),
                                        lambda: (best_nops, best_order))
        return (best_nops, best_order, key), None
    
    out, _ = lax.scan(outer_loop_fn, (10000000.0, best_order, key), (), length=num_iterations)
    return out    

