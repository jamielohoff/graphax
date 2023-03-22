import jax

from graphax.core import front_eliminate, back_eliminate, vertex_eliminate, forward, reverse
from graphax.examples.simple import make_simple
from graphax.examples.helmholtz import make_Helmholtz


edges, info = make_simple()
print(edges)
print(info)

# edges, nops, info = front_eliminate(edges, (1, 2), info)
# print(edges, nops, info)
# gs, nops = front_eliminate(gs, (1,2), gs.info)
# edges, nops = jax.jit(front_eliminate, static_argnums=(1, 2))(edges, (1, 2), info)
# print(edges, nops)
# gs, nops = eliminate(gs, 1, gs.get_info())
# edges, nops1 = vertex_eliminate(edges, 2, info)
# print(edges, nops1)
# edges, nops2 = vertex_eliminate(edges, 1, info)
# print(edges, nops2)
edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
print(edges, nops)

edges, info = make_Helmholtz()
print(edges, info)

edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)

print(edges, nops) # 36 / 56


