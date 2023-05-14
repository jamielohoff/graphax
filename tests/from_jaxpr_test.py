import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph

def f(x, y):
    z = x + y
    w = 4.*z
    u = w + z
    v = 2.*w
    return u, v

print(make_graph(f, 1., 2.))

