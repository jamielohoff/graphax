import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.core import GraphInfo, vertex_eliminate, forward, reverse
from graphax.examples.random import construct_random


key = jrand.PRNGKey(42)
info = GraphInfo(num_inputs=4, 
                num_intermediates=11, 
                num_outputs=4, 
                num_edges=0)
edges, info = construct_random(key, info, fraction=.35)
print(edges)

edges, ops = jax.jit(reverse, static_argnums=(1,))(edges, info)
print(edges, ops)

