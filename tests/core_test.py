import jax

from graphax.core import front_eliminate, back_eliminate, eliminate, forward, reverse
from graphax.examples.simple import construct_simple
from graphax.examples.helmholtz import construct_Helmholtz


edges, info = construct_simple()
print(edges)
info = tuple([int(i) for i in info])
print(info)

# gs, nops = front_eliminate(gs, (2,1), 6, 2)
# gs, nops = front_eliminate(gs, (1,2), gs.info)
# gs, nops = jax.jit(back_eliminate, static_argnums=(1,2))(gs, (1,2), gs.get_info())
# gs, nops = eliminate(gs, 1, gs.get_info())
edges, nops1 = eliminate(edges, 2, info)
edges, nops2 = eliminate(edges, 1, info)
edges, nops = jax.jit(reverse, static_argnums=(1,))(edges, info)

print(nops1 + nops2)
print(edges)

edges, info = construct_Helmholtz()
info = tuple([int(i) for i in info])
print(edges)

edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)
# gs, nops = jax.jit(eliminate, static_argnums=(1,2))(gs, 11, gs.get_info())

print(nops) # 36 / 56
print(edges)


