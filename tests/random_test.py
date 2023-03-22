import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.core import make_graph_info, vertex_eliminate, forward, reverse
from graphax.examples.random import make_random, make_connected_random

# import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

key = jrand.PRNGKey(42)
info = make_graph_info([4, 11, 4])
edges, info = make_random(key, info, fraction=.35)
print(edges)

edges, ops = jax.jit(reverse, static_argnums=(1,))(edges, info)
print(edges, ops)

edges, info = make_connected_random(key, info, p=jnp.array([.4, .5, .1]))
print(edges)

edges, ops = jax.jit(reverse, static_argnums=(1,))(edges, info)
print(edges, ops)

