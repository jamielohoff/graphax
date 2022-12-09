import jax
import jax.numpy as jnp
import jax.random as jrand

from scipy.sparse import lil_matrix

from graphax.elimination import front_eliminate, back_eliminate, eliminate, forward, reverse
from graphax.examples.random import construct_random_graph

key = jrand.PRNGKey(42)
gs = construct_random_graph(4, 11, 4, key)

print(lil_matrix(gs.edges))

# gs, nops = eliminate(gs, 3, 6)
# gs, nops2 = jax.jit(eliminate, static_argnums=2)(gs, 3, 6)
# gs, nops1 = jax.jit(eliminate, static_argnums=2)(gs, 2, 6)
gs, nops = jax.jit(reverse, static_argnums=(1))(gs, gs.get_info())

print(nops)
print(lil_matrix(gs.edges))
print(gs.state)

