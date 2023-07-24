import jax
import jax.random as jrand

from graphax.core import vertex_eliminate, forward, reverse
from graphax.examples import make_Helmholtz
from graphax.transforms.embedding import embed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

key = jrand.PRNGKey(42)
new_info = [3, 7, 1]

edges = make_Helmholtz()
print(edges.shape)
edges = embed(key, edges, new_info)
print(edges)
edges, nops = jax.jit(forward)(edges)
print(edges, nops)

# edges, info = make_lighthouse()
# edges, info = embed(edges, info, new_info)
# print(edges, info)
# edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
# print(nops)

# edges, info = make_adaptive_LIF()
# edges, info = embed(edges, info, new_info)
# print(edges, info)
# edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
# print(nops)

# edges, info = make_free_energy()
# edges, info = embed(edges, info, new_info)
# print(edges, info)
# edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
# print(nops)

