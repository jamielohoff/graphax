from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def make_LIF() -> Tuple[chex.Array, GraphInfo]:
    def lif(U, I, S, a, b, threshold):
        U_next = a*U + (1-a)*I
        I_next = b*I + (1-b)*S
        S_next = jnp.heaviside(U_next - threshold, 0.)
        
        return U_next, I_next, S_next
    
    edges, info = make_graph(lif, .1, .2, 1., .95, .9, 1.)
    return edges, info


# TODO add adaptive threshold
def make_adaptive_LIF() -> Tuple[chex.Array, GraphInfo]:
    def ada_lif(U, I, S, a, b, threshold):
        U_next = a*U + (1-a)*I
        I_next = b*I + (1-b)*S
        S_next = jnp.heaviside(U_next - threshold, 0.)
        
        return U_next, I_next, S_next
    
    edges, info = make_graph(ada_lif, .1, .2, 1., .95, .9, 1.)
    return edges, info

