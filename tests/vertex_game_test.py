from graphax.vertex_game import VertexGame, make_vertex_game_state
from graphax.examples.helmholtz import make_Helmholtz

edges, info = make_Helmholtz()

state = make_vertex_game_state(info, edges)

env = VertexGame(state)
state, reward, terminated = env.step(state, 0)
state, reward, terminated = env.step(state, 1)
state, reward, terminated = env.step(state, 2)
state, reward, terminated = env.step(state, 3)
state, reward, terminated = env.step(state, 4)
state, reward, terminated = env.step(state, 5)
state, reward, terminated = env.step(state, 6)
state, reward, terminated = env.step(state, 7)
state, reward, terminated = env.step(state, 8)
state, reward, terminated = env.step(state, 9)
state, reward, terminated = env.step(state, 10)
print(state.edges, reward, terminated)

