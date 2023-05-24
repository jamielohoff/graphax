from graphax.core import forward_gpu, reverse_gpu
from graphax.examples import make_f, make_g, make_LIF
from graphax.transforms import safe_preeliminations_gpu, compress_graph


edges, info = make_f()
edges, info = safe_preeliminations_gpu(edges, info)
edges, info = compress_graph(edges, info)
print(edges, info)
_, ops = forward_gpu(edges, info)
print(ops)
_, ops = reverse_gpu(edges, info)
print(ops)

edges, info = make_g()
edges, info = safe_preeliminations_gpu(edges, info)
edges, info = compress_graph(edges, info)

print(edges, info)
_, ops = forward_gpu(edges, info)
print(ops)
_, ops = reverse_gpu(edges, info)
print(ops)


edges, info = make_LIF()
edges, info = safe_preeliminations_gpu(edges, info)
edges, info = compress_graph(edges, info)

print(edges, info)
_, ops = forward_gpu(edges, info)
print(ops)
_, ops = reverse_gpu(edges, info)
print(ops)

