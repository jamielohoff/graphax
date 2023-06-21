import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph


def simple(x, y):
    z = x*y
    w = jnp.sin(z)
    return w+z, jnp.log(w)

edges, info, vertex_mask, attn_mask = make_graph(simple, 1., 1.)
print(edges, info, vertex_mask, attn_mask)

def Helmholtz(x):
    e = jnp.sum(x)
    f = 1. + -e
    w = x / f
    z = jnp.log(w)
    return x*z

x = jnp.ones(4)
edges, info, vertex_mask, attn_mask = make_graph(Helmholtz, x)
print(edges, info, vertex_mask, attn_mask)

