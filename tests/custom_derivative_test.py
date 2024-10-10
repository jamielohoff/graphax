import jax
import jax.numpy as jnp

import graphax as gx

@gx.custom_elemental
def g(x):
  return x #2. + jnp.sin(x)/2.

@g.defelemental
def f_elemental(primals):
  return jnp.tanh(primals[0])

def f(x, y):
    z = x*y
    w = jnp.sin(z)
    # w = list(w)[0]
    return z + w, jnp.log(w)

print(jax.make_jaxpr(f)(2., 3.))
g = jax.jit(f)
g(2., 3.)
# print(gx.jacve(f)(2., 3.))
# print("#" * 80)
# print(jax.make_jaxpr(gx.jacve(f, order="rev", argnums=(0, 1)))(2., 3.))

