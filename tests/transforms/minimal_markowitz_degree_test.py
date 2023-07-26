from graphax import forward, reverse, cross_country
from graphax.examples import (make_simple, 
                                make_Helmholtz, 
                                make_transformer_encoder, 
                                make_transformer_encoder_decoder,
                                make_lif_SNN,
                                make_ada_lif_SNN,
                                make_1d_roe_flux,
                                make_6DOF_robot)
from graphax.transforms.markowitz import minimal_markowitz
from graphax.transforms.preelimination import safe_preeliminations
from graphax.transforms.compression import compress_graph

# edges= make_simple()
# order = minimal_markowitz(edges)
# print(order)

# edges = make_Helmholtz()
# order = minimal_markowitz(edges)
# print(order)

# print(cross_country(order, edges)[1])


edges = make_transformer_encoder()
edges = safe_preeliminations(edges)
edges = compress_graph(edges)
order = minimal_markowitz(edges)
print(order)

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


edges = make_ada_lif_SNN()
edges = safe_preeliminations(edges)
edges = compress_graph(edges)
order = minimal_markowitz(edges)
print(order)

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


edges = make_1d_roe_flux()
edges = safe_preeliminations(edges)
edges = compress_graph(edges)
order = minimal_markowitz(edges)
print(order)
print(len(order))

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


edges = make_6DOF_robot()
edges = safe_preeliminations(edges)
edges = compress_graph(edges)
order = minimal_markowitz(edges)
print(order)
print(len(order))

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


edges = make_transformer_encoder_decoder()
edges = safe_preeliminations(edges)
edges = compress_graph(edges)
order = minimal_markowitz(edges)
print(order)
print(len(order))

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

