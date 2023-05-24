from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def make_LIF() -> Tuple[chex.Array, GraphInfo]:
    def lif(U, I):
        pass


def make_adaptive_LIF() -> Tuple[chex.Array, GraphInfo]:
    def ada_lif(U, I):
        pass

