import jax
import jax.numpy as jnp

import graphax as gx
from graphax.examples.general_relativity import metric


g_jac = gx.jacve(metric, order="fwd", argnums=(0, 1, 2, 3))

x = jnp.ones(())
y = jnp.ones(())
z = jnp.ones(())
w = jnp.ones(())

jaxpr = jax.make_jaxpr(metric)(x, y, z, w)
print(jaxpr)
jaxpr = jax.make_jaxpr(g_jac)(x, y, z, w)
print(jaxpr, len(jaxpr.eqns))

edges = gx.make_graph(g_jac, x, y, z, w)
print(edges, edges.shape)
unprocessed_order = gx.minimal_markowitz(edges)

# edges = gx.clean(edges)
# edges = gx.compress(edges)
# print(edges.shape)

# _, ops = gx.forward(edges)
# print("fwd", ops)

# _, ops = gx.reverse(edges)
# print("rev", ops)

# order = gx.minimal_markowitz(edges)
# output, ops = gx.cross_country(order, edges)
# _, ops = output
# print("mM", ops)

g_hess = gx.jacve(g_jac, order=unprocessed_order, argnums=(0, 1, 2, 3))
hessian = g_hess(x, y, z, w)

g_hess_jax = jax.jacfwd(g_jac, argnums=(0, 1, 2, 3))
jax_hessian = g_hess(x, y, z, w)

print(gx.tree_allclose(hessian, jax_hessian))

