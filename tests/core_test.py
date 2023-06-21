import jax

from graphax.core import (front_eliminate, 
                            back_eliminate, 
                            vertex_eliminate, 
                            forward, 
                            reverse)
from graphax.examples import make_simple
from graphax.examples import make_Helmholtz


edges, info, vertex_mask, attn_mask = make_simple()
print(edges, info, vertex_mask)

_edges, fmas = jax.jit(front_eliminate, static_argnums=(0, 2))((1, 2), edges, info)
print(_edges, fmas)

_edges, fmas = jax.jit(back_eliminate, static_argnums=(0, 2))((1, 2), edges, info)
print(_edges, fmas)

edges, fmas = jax.jit(vertex_eliminate, static_argnums=(0, 2))(1, edges, info)
print(edges, fmas)
edges, fmas = jax.jit(vertex_eliminate, static_argnums=(0, 2))(2, edges, info)
print(edges, fmas)

# edges, info = make_Helmholtz()
# print(edges, info)

# edges, nops = jax.jit(forward, static_argnums=(1,))(edges, info)

# print(edges, nops) # 36 / 56


