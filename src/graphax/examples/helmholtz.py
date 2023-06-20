import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def make_Helmholtz(size: int = 4):
    def Helmholtz(x):
        z = jnp.log(x / (1 + -jnp.sum(x)))
        return x * z

    x = jnp.ones(size)
    return make_graph(Helmholtz, x)


def make_free_energy(size: int = 4):
    def free_energy(x):
        z = jnp.log(x / (1 - jnp.sum(x)))
        return jnp.sum(x * z)

    x = jnp.ones(size)
    return make_graph(free_energy, x)

