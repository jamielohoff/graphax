import jax
import jax.random as jrand

from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_Helmholtz
from graphax.examples import make_random_code
from graphax.transforms.cleaner import clean, connectivity_checker


edges = make_Helmholtz()
print(connectivity_checker(edges))

key = jrand.PRNGKey(42)
info = [10, 20, 10]
code, jaxpr = make_random_code(key, info)
edges = make_graph(jaxpr)
print(edges)
edges = clean(edges)

