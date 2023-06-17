import jax
import jax.numpy as jnp

from graphax.transforms import safe_preeliminations_gpu, compress_graph
from graphax.core import reverse_gpu, forward_gpu, vertex_eliminate_gpu
from graphax.interpreter.from_jaxpr import make_graph


def f(x):
    e = jnp.sum(x)
    f = 1. + -e
    w = x / f
    z = jnp.log(w)
    return x*z

edges, info, output_vertices, attn_mask = make_graph(f, jnp.ones(4))
print(edges, info, output_vertices, attn_mask)
edges, info, output_vertices, attn_mask = safe_preeliminations_gpu(edges, info, output_vertices, attn_mask)
print(edges, info, output_vertices, attn_mask)
edges, info, output_vertices, attn_mask = compress_graph(edges, info, output_vertices, attn_mask)
print(edges, info, output_vertices, attn_mask)
ops = 0

for i in sorted([3, 4, 5, 6, 2, 1])[::-1]:
    edges, nops = vertex_eliminate_gpu(i, edges, info)
    print(i, "###")
    print(edges)
    ops += nops
    print(nops)

print(edges, ops)

# def g(x):
#     return jnp.sum(jnp.sin(x) * jnp.cos(x**2) + jnp.log(x) - x**3 + jnp.exp(x), axis=0)

# print(make_graph(f, jnp.ones(4)))

