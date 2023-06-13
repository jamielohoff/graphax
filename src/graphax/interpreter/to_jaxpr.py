import jax
import jax.numpy as jnp
from jax.core import (ClosedJaxpr, 
                    Jaxpr, 
                    JaxprEqn, 
                    Primitive, 
                    eval_jaxpr, 
                    new_jaxpr_eqn, 
                    Var, 
                    Atom, 
                    Literal, 
                    ShapedArray)
from jax import lax
from jax._src.util import safe_map
from jax._src.interpreters.ad import get_primitive_transpose, primitive_jvps
from jax._src.interpreters.ad import primitive_jvps

elemental_registry = {}

elemental_registry[lax.exp_p] = lax.exp_p
elemental_registry[lax.sin_p] = lax.cos_p
elemental_registry[lax.add_p] = lax.cos_p


def f(x, y):
    return x**2 + y**2

jaxpr = jax.make_jaxpr(f)(1., 1.)

def add_eqn(jaxpr: ClosedJaxpr, eqn: JaxprEqn) -> Jaxpr:
    prim = lax.add_p
    print(primitive_jvps[prim])
    outvar_list = [eqn.outvars[0] for eqn in jaxpr.jaxpr._eqns]
    print(outvar_list)
    
    outvars = [Var(6, "", ShapedArray((), jnp.float32))]
    eqn = new_jaxpr_eqn(outvar_list[1:], outvars, prim, {}, set(), source_info=None)
    print(eqn.source_info)
    jaxpr.jaxpr._eqns.append(eqn)
    
    new_outvar_list = [eqn.outvars[0] for eqn in jaxpr.eqns]
    jaxpr.jaxpr._outvars.append(new_outvar_list[-1])
    return jaxpr

new_jaxpr = add_eqn(jaxpr, None)
print(new_jaxpr)

print(eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.literals, 1., 1.))

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

