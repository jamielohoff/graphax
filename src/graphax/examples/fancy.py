from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def make_f() -> Tuple[chex.Array, GraphInfo]:
    def f(x):
        a = x[0] * x[1]
        b = x[2] - x[3]
        c = jnp.sin(x[0]) + jnp.cos(x[1])
        d = jnp.exp(x[2]) / x[3]
        e = a + b
        f = jnp.sum(x) + c
        g = jnp.log(d)
        h = jnp.sqrt(jnp.abs(e))
        i = jnp.tan(h)
        j = jnp.maximum(x, 0)
        return jnp.array([e, f, g, i])

    x = jnp.ones(4)
    edges, info = make_graph(f, x)
    return edges, info


def make_g(size: int = 5) -> Tuple[chex.Array, GraphInfo]:
    def g(x):
        return jnp.sum(jnp.sin(x) * jnp.cos(x**2) + jnp.log(x) - x**3 + jnp.exp(x), axis=0)
    
    x = jnp.ones(size)
    edges, info = make_graph(g, x)
    return edges, info

