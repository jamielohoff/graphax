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
print(jax.make_jaxpr(Helmholtz)(x))
edges, info, vertex_mask, attn_mask = make_graph(Helmholtz, x)
print(edges, info, vertex_mask, attn_mask)


def f(x, y):
    return 2.* x * y

x = jnp.ones(4)
f_jac = jax.jacrev(f, argnums=(0,1))
print(jax.make_jaxpr(f_jac)(x, x))
print(f_jac(x, x))
edges, info, vertex_mask, attn_mask = make_graph(f, x, x)
print(edges, info, vertex_mask, attn_mask)

