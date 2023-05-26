import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph
from graphax.core import reverse_gpu, vertex_eliminate_gpu
from graphax.transforms.preelimination import safe_preeliminations_gpu, compress_graph


def Helmholtz(x):
    z = jnp.log(x / (1 - jnp.sum(x)))
    return x * z

x = jnp.ones(4)
edges, info = make_graph(Helmholtz, x)
print(edges, info)

edges, info = safe_preeliminations_gpu(edges, info)
print(edges, info)
edges, info = compress_graph(edges, info)
print(edges, info)

def f(x, y):
    z = x + y
    a = jnp.arctan(.33)
    b = jnp.arccos(.25)
    c = a + b
    w = jnp.cos(z/c)
    return w + z, 2.*w

x = jnp.ones(2)
edges, info = make_graph(f, 1., 1.)
print(edges, info)

edges, info = safe_preeliminations_gpu(edges, info)
print(edges, info)
edges, info = compress_graph(edges, info)
print(edges, info)
