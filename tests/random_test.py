import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.core import make_graph_info, vertex_eliminate, forward, reverse
from graphax.examples.random import construct_random


key = jrand.PRNGKey(42)
info = make_graph_info([4, 11, 4])
edges, info = construct_random(key, info, fraction=.35)
print(edges)

edges, ops = jax.jit(reverse, static_argnums=(1,))(edges, info)
print(edges, ops)

