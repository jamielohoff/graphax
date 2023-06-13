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


def make_minimal_reverse() -> Tuple[chex.Array, GraphInfo]:
    # Define the component functions
    def g1(x):
        return x[0] ** 2 + x[1] ** 2

    def g2(x):
        return jnp.sin(x[2]) + jnp.log(x[3])
    
    # Define the overall function
    def minimal_reverse(x):
        return g1(x) + g2(x)

    x = jnp.ones(4)
    edges, info = make_graph(minimal_reverse, x)
    return edges, info


def make_hessian() -> Tuple[chex.Array, GraphInfo]:
    def f(x):
        z = jnp.cos(x[1]) * jnp.sin(x[2])
        w = jnp.exp(x[0] + x[3])
        return z + w

    x = jnp.ones(4)
    grad_f = jax.grad(f)
    edges, info = make_graph(grad_f, x)
    return edges, info


def make_softmax_attention():
    def attn(q, k, v):
        a = jnp.outer(q, k)
        b = jnp.softmax(a, axis=-1)
        return b*v
    
    q = jnp.ones(3)
    k = jnp.ones(3)
    v = jnp.ones(3)
    
    edges, info = make_graph(attn, q, k, v)
    return edges, info

