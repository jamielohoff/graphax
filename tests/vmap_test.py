import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose
from graphax.examples import RobotArm_6DOF


def f(x, y):
    z = x @ y
    w = z**3
    return w + z, 5*w

x = jnp.ones((50, 3, 3))
y = jnp.ones((50, 3, 3))
jaxpr = jax.make_jaxpr(f)(x, y)
print(jaxpr)
vmap_f = jax.vmap(f, in_axes=(0, 0))

jacrev_f = jax.jit(jacve(vmap_f, order="rev", argnums=(0, 1), count_ops=True))
jaxpr = jax.make_jaxpr(jacrev_f)(x, y)
print(jaxpr)
veres = jacrev_f(x, y)
print(veres)

jac_f = jax.jit(jax.jacrev(vmap_f, argnums=(0, 1)))
revres = jac_f(x, y)
print(revres)
print(tree_allclose(veres, revres))



# shape = (60,) 
# xs = [jnp.ones(shape)*0.002]*6
# jaxpr = jax.make_jaxpr(position_angles_6DOF)(*xs)
# print(jaxpr)

# jacrev_f = jax.jit(jacve(jax.vmap(position_angles_6DOF, in_axes=(0, 0, 0, 0, 0, 0)), order="rev", argnums=(0, 1), count_ops=False))
# jaxpr = jax.make_jaxpr(jacrev_f)(*xs)
# print(jaxpr)
# veres = jacrev_f(*xs)
# print(veres)

# jac_f = jax.jit(jax.jacrev(jax.vmap(position_angles_6DOF, in_axes=(0, 0, 0, 0, 0, 0)), argnums=(0, 1)))
# revres = jac_f(*xs)
# print(revres)
# print(tree_allclose(veres, revres))

