from graphax import forward, reverse, cross_country, get_shape
from graphax.examples import (make_simple, 
                                make_Helmholtz, 
                                make_transformer_encoder, 
                                make_transformer_encoder_decoder,
                                make_ada_lif_SNN,
                                make_lif_SNN,
                                make_1d_roe_flux,
                                make_3d_roe_flux,
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

edges, jaxpr = make_lif_SNN()
print(edges)
print(jaxpr)
print(len(jaxpr.eqns))
num_i, num_v = get_shape(edges)
print(edges.at[0, 0, 0:3].get())
_, preelim = safe_preeliminations(edges, return_preeliminated=True)
print(preelim)
# for v in preelim:
#     print(v, edges[0, 1:, v-1].sum(), edges[0, num_i+v, :].sum())
#     print(jaxpr.eqns[v-1].outvars)
#     print(jaxpr.eqns[v-1])

order = minimal_markowitz(edges)
print(order)
print(edges.shape[-1])
print(len(order))
print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

print("###")

edges = safe_preeliminations(edges)
edges = compress_graph(edges)
print(edges.at[0, 0, 0:3].get())
order = minimal_markowitz(edges)
print(order)
print(edges.shape[-1])
print(len(order))

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


# edges = make_3d_roe_flux()
# edges = safe_preeliminations(edges)
# edges = compress_graph(edges)
# order = minimal_markowitz(edges)
# print(order)
# print(len(order))

# print(forward(edges)[1])
# print(reverse(edges)[1])
# print(cross_country(order, edges)[1])


# edges = make_6DOF_robot()
# edges = safe_preeliminations(edges)
# edges = compress_graph(edges)
# order = minimal_markowitz(edges)
# print(order)
# print(len(order))

# print(forward(edges)[1])
# print(reverse(edges)[1])
# print(cross_country(order, edges)[1])

