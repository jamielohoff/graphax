from typing import Callable, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src import custom_api_util


@custom_api_util.register_custom_decorator_type
class custom_elemental:
    fn: Callable
    nondiff_argnums: Sequence[int]
    elemental_partial: Callable
    
    def __init__(self, fn, nondiff_argnums):
        self.fn = fn
        self.nondiff_argnums = nondiff_argnums
        self.elemental_partial = lambda x: x
        
    def defelemental(self, elemental_partial):
        self.elemental_partials = elemental_partial
        return elemental_partial
    
    def elemental(primals, *params):
        pass

