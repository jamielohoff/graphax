import jax
import jax.numpy as jnp

from graphax.vertex_game import VertexGame
from graphax.examples.simple import construct_simple

gs = construct_simple()

env = VertexGame(gs)

gs, rew2, done = env.step(gs, 1)
print(gs.edges)
gs, rew1, done = env.step(gs, 0)

print(rew1 + rew2)
print(gs.edges)
print(gs.state)
print(done)

