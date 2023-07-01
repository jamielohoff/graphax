from graphax.core import forward, reverse
from graphax.examples import (make_simple,
                            make_f, 
                            make_g, 
                            make_minimal_reverse,
                            make_LIF, 
                            make_adaptive_LIF, 
                            make_Helmholtz, 
                            make_lighthouse, 
                            make_hole, 
                            make_scalar_assignment_tree,
                            make_sdf_sphere,
                            make_sdf_box,
                            make_hessian,
                            make_softmax_attention)


edges = make_softmax_attention()
_, ops = forward(edges)
print(ops)
_, ops = reverse(edges)
print(ops)


# edges, info = make_LIF()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)


# edges, info = make_sdf_sphere()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)


# edges, info = make_sdf_box()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)


# edges, info = make_hessian()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)

