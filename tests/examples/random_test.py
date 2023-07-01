import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.examples import make_random_jaxpr


key = jrand.PRNGKey(42)
jaxpr = make_random_jaxpr(key, [4, 11, 4], fraction=.45)

