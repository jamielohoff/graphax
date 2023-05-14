import jax
import jax.numpy as jnp
import time
from graphax.core import (vertex_eliminate, 
                          vertex_eliminate_gpu, 
                          make_graph_info,
                          reverse,
                          reverse_gpu)
from graphax.examples import make_random, make_simple, make_Helmholtz

cpu_devices = jax.devices("cpu")

reverse = jax.jit(reverse, static_argnums=(1,))
reverse_gpu = jax.jit(reverse_gpu, static_argnums=(1,))

info = make_graph_info([25, 50, 5])
edges, info = make_random(jax.random.PRNGKey(42), info, fraction=0.9)
print(edges.device())

st = time.time()
new_edges, nops = reverse(jax.device_put(edges, device=cpu_devices[0]), info)
print(time.time() - st)
print(new_edges, nops)

info = make_graph_info([25, 50, 5])
edges, info = make_random(jax.random.PRNGKey(42), info, fraction=0.9)
st = time.time()
new_edges, nops = reverse_gpu(jax.device_put(edges, device=cpu_devices[0]), info)
print(time.time() - st)
print(new_edges, nops)

