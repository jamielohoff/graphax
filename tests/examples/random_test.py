import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.core import make_graph_info, vertex_eliminate, forward_gpu, reverse_gpu
from graphax.examples import make_random, make_connected_random


key = jrand.PRNGKey(42)
info = make_graph_info([4, 11, 4])
edges, info, vertex_mask, attn_mask = make_random(key, info, fraction=.45)
print(edges, vertex_mask)

edges, ops = jax.jit(reverse_gpu, static_argnums=(2,))(edges, info, vertex_mask)
print(edges, ops)

edges, info, vertex_mask, attn_mask = make_connected_random(key, info, p=jnp.array([.2, .5, .15, .15]))
print(edges, vertex_mask)

edges, ops = jax.jit(forward_gpu, static_argnums=(2,))(edges, info, vertex_mask)
print(edges, ops)

