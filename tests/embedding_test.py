import jax
import jax.random as jrand

from graphax.core import front_eliminate, back_eliminate, vertex_eliminate, forward, reverse
from graphax.examples import make_Helmholtz, make_adaptive_LIF, make_lighthouse, make_free_energy
from graphax.core import make_graph_info
from graphax.transforms.embedding import embed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

key = jrand.PRNGKey(42)
new_info = make_graph_info([10, 15, 5])

edges, info = make_Helmholtz()
edges, info = embed(edges, info, new_info)
print(edges, info)
edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
print(nops)

edges, info = make_lighthouse()
edges, info = embed(edges, info, new_info)
print(edges, info)
edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
print(nops)

edges, info = make_adaptive_LIF()
edges, info = embed(edges, info, new_info)
print(edges, info)
edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
print(nops)

edges, info = make_free_energy()
edges, info = embed(edges, info, new_info)
print(edges, info)
edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
print(nops)

