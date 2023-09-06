from graphax.core import forward, reverse
from graphax.examples import (make_simple,
                            make_f,  
                            make_Helmholtz, 
                            make_lighthouse, 
                            make_hole, 
                            make_scalar_assignment_tree)


edges = make_Helmholtz()
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


# edges, info = make_hessian()
# edges, info = safe_preeliminations_gpu(edges, info)
# edges, info = compress_graph(edges, info)

# print(edges, info)
# _, ops = forward_gpu(edges, info)
# print(ops)
# _, ops = reverse_gpu(edges, info)
# print(ops)

