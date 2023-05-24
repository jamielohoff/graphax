from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def make_Helmholtz(size: int = 4) -> Tuple[chex.Array, GraphInfo]:
    def Helmholtz(x):
        z = jnp.log(x / (1 - jnp.sum(x)))
        return x * z

    x = jnp.ones(size)
    edges, info = make_graph(Helmholtz, x)
    return edges, info


def make_free_energy(size: int = 4) -> Tuple[chex.Array, GraphInfo]:
    def free_energy(x):
        z = jnp.log(x / (1 - jnp.sum(x)))
        return jnp.sum(x * z)

    x = jnp.ones(size)
    edges, info = make_graph(free_energy, x)
    return edges, info

