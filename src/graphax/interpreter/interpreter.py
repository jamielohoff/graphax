import numpy as np
from functools import wraps

import jax
from jax import core
from jax import lax
from jax._src.util import safe_map

elemental_registry = {}

elemental_registry[lax.exp_p] = lax.exp_p
elemental_registry[lax.sin_p] = lax.cos_p

def elementals(f):
    @wraps
    def wrapped(*args, **kwargs):
        closed_jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        out = jaxpr_with_elementals(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out[0]
    return wrapped


def jaxpr_with_elementals():

