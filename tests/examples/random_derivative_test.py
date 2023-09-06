import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx
from graphax.examples import make_random_derivative_code


info = [5, 10, 1]
key = jrand.PRNGKey(1234)
code, jaxpr = make_random_derivative_code(key, info, primal_p=jnp.array([.2, .8, 0.]), primitive_p=jnp.array([0.15, 0.75, 0.05, 0.05, 0.]))
print(code)
print(jaxpr)

edges = gx.make_graph(jaxpr)
print(edges)
edges, order = gx.safe_preeliminations(edges, return_order=True)
edges = gx.compress(edges)

print(gx.forward(edges)[1])
print(gx.reverse(edges)[1])
order = gx.minimal_markowitz(edges)
print(gx.cross_country(order, edges)[1])




