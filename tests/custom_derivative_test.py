import jax
import jax.numpy as jnp

import graphax as gx
from graphax.sparse.custom_derivatives import custom_elemental_p

# @gx.custom_elemental
# def g(x):
#   return x #2. + jnp.sin(x)/2.

# @g.defelemental
# def f_elemental(primals):
#   return jnp.tanh(primals[0])

def f(x, y):
    z = x*y
    w = custom_elemental_p.bind(z)
    # w = list(w)[0]
    return z + w, jnp.log(w)

print(jax.jit(f)(2., 3.))
# print(gx.jacve(f)(2., 3.))
# print("#" * 80)
# print(jax.make_jaxpr(gx.jacve(f, order="rev", argnums=(0, 1)))(2., 3.))

