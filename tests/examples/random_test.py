import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples.random import make_random_jaxpr


info = [2, 10, 1]
key = jrand.PRNGKey(42)
print(make_random_jaxpr(key, info))

