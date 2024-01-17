import graphax as gx
from graphax.examples import (make_simple,
                            make_f,  
                            make_Helmholtz, 
                            make_lighthouse, 
                            make_hole,
                            make_cloud_schemes,
                            make_HumanHeartDipole,
                            make_PropaneCombustion,
                            make_Kerr_Sen_metric,
                            make_BlackScholes,
                            make_BlackScholes_Jacobian)


edges = make_BlackScholes_Jacobian()
print(edges)
print(gx.get_shape(edges))
_, ops = gx.forward(edges)
print(ops)
_, ops = gx.reverse(edges)
print(ops)

order = gx.minimal_markowitz(edges)
output, ops = gx.cross_country(order, edges)
_, ops = output
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

