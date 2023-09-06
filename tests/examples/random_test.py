import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx
from graphax.examples.random_codegenerator import make_random_code


info = [15, 105, 20]
key = jrand.PRNGKey(123)
code, jaxpr = make_random_code(key, info, primal_p=[1, 0, 0], primitive_p=[.2, .8, .0, .0, .0])
print(code)
print(jaxpr)

edges = gx.make_graph(jaxpr)
print(edges)

edges = gx.make_graph(jaxpr)
# print(edges)
# edges = gx.safe_preeliminations(edges)
# edges = gx.compress(edges)

_, fops = gx.forward(edges)
_, rops = gx.reverse(edges)
order = gx.minimal_markowitz(edges)
_, ccops = gx.cross_country(order, edges)
print(fops, rops, ccops, f"gain: {100.*(1.-ccops/min(fops, rops)):.2f}%")


