import jax
import jax.numpy as jnp

from graphax.vertex_game import EdgeGame
from graphax.examples.simple import construct_simple

gs = construct_simple()

env = EdgeGame(gs)

print(gs.edges)

new_gs, rew2, done = env.step(gs, jnp.array([2, 1, 0]))
print(new_gs.edges, rew2)
# gs, rew1, done = env.step(gs, ())

# print(rew1 + rew2)
# print(gs.edges)
# print(gs.state)
# print(done)

