import jax

from graphax.jax.elimination import front_eliminate, back_eliminate, eliminate, forward, reverse
from graphax.jax.examples.simple import construct_simple
from graphax.jax.examples.helmholtz import construct_Helmholtz


gs = construct_simple()

print(gs.edges)

# gs, nops = front_eliminate(gs, (2,1), 6, 2)
# gs, nops = front_eliminate(gs, (1,2), gs.info)
# gs, nops = jax.jit(back_eliminate, static_argnums=(1,2))(gs, (1,2), gs.get_info())
# gs, nops = eliminate(gs, 1, gs.get_info())
gs, nops1 = eliminate(gs, 1, gs.get_info())
gs, nops2 = eliminate(gs, 2, gs.get_info())
gs, nops = jax.jit(reverse, static_argnums=(1,))(gs, gs.get_info())

print(nops1 + nops2)
print(gs.edges)
print(gs.state)
print(gs.info)

gs = construct_Helmholtz()

print(gs.edges)

gs, nops = jax.jit(reverse, static_argnums=(1,))(gs, gs.get_info())
# gs, nops = jax.jit(eliminate, static_argnums=(1,2))(gs, 11, gs.get_info())

print(nops) # 36 / 56
print(gs.edges)
print(gs.state)
print(gs.info)

