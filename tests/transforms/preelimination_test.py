import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples import make_Helmholtz, make_random_code
from graphax.transforms.preelimination import safe_preeliminations
from graphax.interpreter.from_jaxpr import make_graph

import sys
jnp.set_printoptions(threshold=sys.maxsize)

edges = make_Helmholtz()

edges = safe_preeliminations(edges)
print(edges)

key = jrand.PRNGKey(42)
info = [5, 15, 5]
code, jaxpr = make_random_code(key, info)
print(jaxpr)
edges = make_graph(jaxpr)
print(edges)
edges = safe_preeliminations(edges)

