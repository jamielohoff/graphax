import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph
from graphax.core import safe_pre_eliminations_gpu, reverse_gpu


def Helmholtz(x):
    z = jnp.log(x / (1 - jnp.sum(x)))
    return x * z

x = jnp.ones(4)
edges, info = make_graph(Helmholtz, x)
print(reverse_gpu(edges, info))

edges, info, ops = safe_pre_eliminations_gpu(edges, info)
print(edges, info, ops)

print(reverse_gpu(edges, info))

