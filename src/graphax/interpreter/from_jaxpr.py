from typing import Callable, Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo, make_empty_edges, make_graph_info

def f(x, y):
    z = x + y
    w = 4.*z
    u = w + z
    v = 2.*w
    return u, v

f_jaxpr = jax.make_jaxpr(f)(1., 2.)
print("literals:", f_jaxpr.literals)
print(f_jaxpr.in_avals)
print(f_jaxpr.eqns)
print(f_jaxpr.out_avals)

def make_graph(f: Callable, *x: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    print()
    f_jaxpr = jax.make_jaxpr(f)(*x)
    
    num_i = len(f_jaxpr.in_avals)
    num_v = len(f_jaxpr.eqns)
    num_o = len(f_jaxpr.out_avals)
    
    info = make_graph_info([num_i, num_v, num_o])
    edges = make_empty_edges(info)
    
    return edges, info


