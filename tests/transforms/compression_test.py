import jax
import jax.random as jrand

from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_random_code
from graphax.transforms.clean import clean
from graphax.transforms.compression import compress


key = jrand.PRNGKey(42)
info = [10, 20, 10]
code, jaxpr = make_random_code(key, info)
edges = make_graph(jaxpr)
print(edges)
edges = clean(edges)
print(edges.shape)

edges = compress(edges)
print(edges)
print(edges.shape)

