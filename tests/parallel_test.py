import jax
import jax.numpy as jnp
import time
from graphax.core import vertex_eliminate, vertex_eliminate_gpu, make_graph_info
from graphax.examples import make_random, make_simple, make_Helmholtz


vertex_eliminate = jax.jit(vertex_eliminate, static_argnums=(2,))
# vertex_eliminate_gpu = jax.jit(vertex_eliminate_gpu, static_argnums=(2,))

info = make_graph_info([10, 15, 5])
edges, info = make_random(jax.random.PRNGKey(42), info, fraction=0.9)
edges, nops1 = vertex_eliminate(edges, 2, info)
st = time.time()
edges, nops2 = vertex_eliminate(edges, 1, info)
print(time.time() - st)
print(edges, nops1 + nops2)

edges, info = make_random(jax.random.PRNGKey(42), info, fraction=0.9)
edges, nops1 = vertex_eliminate_gpu(edges, 2, info)
st = time.time()
edges, nops2 = vertex_eliminate_gpu(edges, 1, info)
print(time.time() - st)
print(edges, nops1 + nops2)

