import graphax as gx
from graphax.examples import (make_simple, 
                                make_Helmholtz, 
                                make_transformer_encoder, 
                                make_transformer_encoder_decoder,
                                make_ada_lif_SNN,
                                make_lif_SNN,
                                make_1d_roe_flux,
                                make_3d_roe_flux,
                                make_6DOF_robot,
                                make_cloud_schemes,
                                make_Kerr_Sen_metric,
                                make_f)


# edges= make_simple()
# order = minimal_markowitz(edges)
# print(order)

# edges = make_Helmholtz()
# order = minimal_markowitz(edges)
# print(order)

# print(cross_country(order, edges)[1])

edges = make_f()
print(edges)
num_i, num_v = gx.get_shape(edges)
# print(edges.at[0, 0, 0:3].get())
# print(jaxpr.eqns[31].outvars)
# print(edges[0, num_i+32, :], edges[0, :, 31], edges[0, 35, 31])
_, preelim_order = gx.safe_preeliminations(edges, return_order=True)
order = gx.minimal_markowitz(edges)
_, preelim_ops = gx.cross_country(preelim_order, edges)
print("preelimination ops", preelim_ops)
# import jax.numpy as jnp
# for v in order:
#     print(v, jnp.where(edges[0, 1:, v-1] > 0, 1, 0).sum(), jnp.where(edges[0, num_i+v, :] > 0, 1, 0).sum())

print("pre", preelim_order)
print("mMd", order)

print(len(order))
print(gx.forward(edges)[1])
print(gx.reverse(edges)[1])
print(gx.cross_country(order, edges)[1])

print("###")

edges = gx.safe_preeliminations(edges)
edges = gx.compress(edges)
print(edges.at[0, 0, 0:3].get())
order = gx.minimal_markowitz(edges)
print(order)
print(edges.shape[-1])
print(len(order))

print(gx.forward(edges)[1])
print(gx.reverse(edges)[1])
print(gx.cross_country(order, edges)[1])


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

