import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples.random_codegenerator import make_random_code
from graphax.interpreter.from_jaxpr import make_graph


info = [20, 50, 20]
key = jrand.PRNGKey(1234)
code, jaxpr = make_random_code(key, info)
print(code)
print(jaxpr)

edges = make_graph(jaxpr)
print(edges)

