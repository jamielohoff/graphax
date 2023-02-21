import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.core import front_eliminate, back_eliminate, eliminate, forward, reverse
from graphax.examples.random import construct_random_graph
from graphax.examples.helmholtz import construct_Helmholtz


key = jrand.PRNGKey(42)
gs = construct_random_graph(4, 11, 4, key, fraction=.35)
print(gs.edges)

# gs, nops = eliminate(gs, 3, 6)
# gs, nops2 = jax.jit(eliminate, static_argnums=2)(gs, 3, 6)
# gs, nops1 = jax.jit(eliminate, static_argnums=2)(gs, 2, 6)
new_gs, ops = jax.jit(reverse, static_argnums=(1,))(gs, gs.get_info())

print(ops)
print(new_gs.state)
print(new_gs.edges)

