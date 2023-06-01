import jax
import jax.nn as jnn

from graphax.vertex_game import VertexGame, make_vertex_game_state
from graphax.examples import make_Helmholtz

batched_one_hot = jax.vmap(jnn.one_hot, in_axes=(0, None))

edges, info = make_Helmholtz()
print(info)
state = make_vertex_game_state(edges, info)

env = VertexGame(info)
for i in range(0, 14):
    obs, state, reward, terminated = env.step(state, i)
    print(state.vertices)
    one_hot = batched_one_hot(state.vertices-1, info.num_intermediates)
    print(one_hot)
# print(obs)
# print(state.edges)
# print(state.attn_mask)
# print(reward, terminated)

