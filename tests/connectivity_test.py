import jax
import jax.random as jrand

from graphax.core import front_eliminate, back_eliminate, vertex_eliminate, forward, reverse
from graphax.examples import make_Helmholtz
from graphax.examples import make_random
from graphax.transforms.checker import connectivity_checker
from graphax.transforms.cleaner import clean


edges, info = make_Helmholtz()
print(connectivity_checker(edges, info))

key = jrand.PRNGKey(42)
edges, info = make_random(key, info, fraction=.2)
print(edges)
edges, info = clean(edges, info)



