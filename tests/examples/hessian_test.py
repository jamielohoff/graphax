import jax
import jax.numpy as jnp

import graphax as gx

# import sys
# import jax.numpy as jnp
# jnp.set_printoptions(threshold=sys.maxsize)


def f(x, y):
    z = x + y
    w = jnp.sin(z)
    return jnp.log(w), z*w

f_jac = jax.jacrev(f, argnums=(0, 1))

jaxpr = jax.make_jaxpr(f_jac)(1., 1.)
print(jaxpr)

edges = gx.make_graph(f_jac, 1., 1.)
print(edges, edges.shape)
# edges = gx.clean(edges)
# edges = gx.compress(edges)
print(edges.shape)

_, ops = gx.forward(edges)
print(ops)

_, ops = gx.reverse(edges)
print(ops)

