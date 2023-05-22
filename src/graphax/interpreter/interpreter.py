import numpy as np
from functools import wraps

import jax
from jax import core
from jax import lax
from jax._src.util import safe_map

elemental_registry = {}

elemental_registry[lax.exp_p] = lax.exp_p
elemental_registry[lax.sin_p] = lax.cos_p

# def elementals(f):
#     @wraps
#     def wrapped(*args, **kwargs):
#         closed_jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
#         out = jaxpr_with_elementals(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
#         return out[0]
#     return wrapped


# How to add a primitive to a jaxpr (not very useful here because it changes jaxpr)
# prim = Primitive("add")
# ins = [Literal(0., ShapedArray((), jnp.float32)), f_jaxpr.jaxpr._outvars[-1]]
# outs = [Var(14, "", ShapedArray((), jnp.float32))]
# e = JaxprEqn(ins, outs, prim, {}, set(), f_jaxpr.eqns[0].source_info)
# f_jaxpr.eqns.append(e)
# f_jaxpr.jaxpr._outvars = f_jaxpr.jaxpr._outvars[:-1] + outs
# print(f_jaxpr)

