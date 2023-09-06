import jax
import jax.numpy as jnp

import graphax as gx
from graphax.examples.general_relativity import g

# import sys
# import jax.numpy as jnp
# jnp.set_printoptions(threshold=sys.maxsize)


def f(x, y):
    z = x + y
    w = jnp.sin(z)
    return jnp.log(w), z*w

g_jac = gx.jacve(g, order="fwd", argnums=(0, 1, 2, 3))
# f_jac = jax.jacrev(f, argnums=(0, 1))

x = 1. # jnp.ones(2)
y = 1. # jnp.ones(2)
z = 1.
w = 1.
jaxpr = jax.make_jaxpr(g)(x, y, z, w)
print(jaxpr)
jaxpr = jax.make_jaxpr(g_jac)(x, y, z, w)
print(jaxpr, len(jaxpr.eqns))

edges = gx.make_graph(g_jac, x, y, z, w)
print(edges, edges.shape)
# edges = gx.clean(edges)
# edges = gx.compress(edges)
print(edges.shape)

_, ops = gx.forward(edges)
print(ops)

_, ops = gx.reverse(edges)
print(ops)

order = gx.minimal_markowitz(edges)
_, ops = gx.cross_country(order, edges)
print(ops)

