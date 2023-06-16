from typing import Tuple

import jax
import jax.nn as jnn
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
    return make_graph(f, x)


def make_g(size: int = 5) -> Tuple[chex.Array, GraphInfo]:
    def g(x):
        return jnp.sum(jnp.sin(x) * jnp.cos(x**2) + jnp.log(x) - x**3 + jnp.exp(x), axis=0)
    
    x = jnp.ones(size)
    return make_graph(g, x)


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
    return make_graph(minimal_reverse, x)


def make_hessian() -> Tuple[chex.Array, GraphInfo]:
    def f(x):
        z = jnp.cos(x[1]) * jnp.sin(x[2])
        w = jnp.exp(x[0] + x[3])
        return z + w

    x = jnp.ones(4)
    grad_f = jax.grad(f)
    return make_graph(grad_f, x)


def make_softmax_attention():
    def softmax(x):
        exp = jnp.exp(x)
        norm = jnp.sum(exp, axis=-1)
        return exp / norm
    
    def attn(q, k, v):
        a1 = q[0] * k
        a2 = q[1] * k
        b1 = jnp.dot(softmax(a1), v)
        b2 = jnp.dot(softmax(a2), v)
        return jnp.array([b1, b2])
    
    q = jnp.ones(2)
    k = jnp.ones(2)
    v = jnp.ones(2)
    
    print(jax.make_jaxpr(attn)(q, k, v))
    
    return make_graph(attn, q, k, v)

