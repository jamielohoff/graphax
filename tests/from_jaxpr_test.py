import jax
import jax.numpy as jnp

from graphax.core import reverse_gpu, forward_gpu
from graphax.interpreter.from_jaxpr import make_graph


def f(x):
    a = x[0] * x[1]
    b = x[2] - x[3]
    c = jnp.sin(x[0]) + jnp.cos(x[1])
    d = jnp.exp(x[2]) / x[3]
    e = a + b
    f = jnp.sum(x) + c
    g = jnp.log(d)
    h = jnp.sqrt(jnp.abs(e))
    i = jnp.arctan(.5)
    j = jnp.tan(h/i)
    k = jnp.maximum(x, 0)
    return jnp.array([e, f, g, j])

print(make_graph(f, jnp.ones(4)))

def g(x):
    return jnp.sum(jnp.sin(x) * jnp.cos(x**2) + jnp.log(x) - x**3 + jnp.exp(x), axis=0)

print(make_graph(g, jnp.ones(4)))

