from graphax.vertex_game import VertexGame, make_vertex_game_state
from graphax.examples import make_Helmholtz

edges, info = make_Helmholtz()
print(info)
state = make_vertex_game_state(edges, info)

env = VertexGame(state)
obs, state, reward, terminated = env.step(state, 0)
obs, state, reward, terminated = env.step(state, 1)
obs, state, reward, terminated = env.step(state, 2)
obs, state, reward, terminated = env.step(state, 3)
obs, state, reward, terminated = env.step(state, 4)
obs, state, reward, terminated = env.step(state, 5)
obs, state, reward, terminated = env.step(state, 6)
obs, state, reward, terminated = env.step(state, 7)
obs, state, reward, terminated = env.step(state, 8)
obs, state, reward, terminated = env.step(state, 9)
obs, state, reward, terminated = env.step(state, 10)
obs, state, reward, terminated = env.step(state, 11)
obs, state, reward, terminated = env.step(state, 12)
obs, state, reward, terminated = env.step(state, 13)
print(obs)
print(state.edges)
print(state.attn_mask)
print(reward, terminated)

