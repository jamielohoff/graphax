import jax
import jax.random as jrand

from graphax.core import front_eliminate, back_eliminate, vertex_eliminate, forward, reverse
from graphax.examples import make_Helmholtz
from graphax.core import make_graph_info
from graphax.embedding import embed

key = jrand.PRNGKey(42)
edges, info = make_Helmholtz()
new_info = make_graph_info([6, 13, 6])

edges, info = embed(key, edges, info, new_info)
print(edges, info)

