import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def make_f():
    def f(x, y, z, w):
        a = x * y
        b = z - w
        c = jnp.sin(x) + jnp.cos(y)
        d = jnp.exp(z) / w
        e = a + b
        f = jnp.sum(x) + c
        g = jnp.log(d)
        h = jnp.sqrt(jnp.abs(e))
        i = jnp.tan(h)
        j = jnp.maximum(x, 0)
        return jnp.sinh(e), -f, 3.*g, i**2

    return make_graph(f, 1., 1., 1., 1.)


def make_hessian():
    def f(x, y, z, w):
        a = jnp.cos(y) * jnp.sin(z)
        b = jnp.exp(x + w)
        return a + b

    grad_f = jax.grad(f)
    return make_graph(grad_f, 1., 1., 1., 1.)

