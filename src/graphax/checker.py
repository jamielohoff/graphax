from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from .core import GraphInfo

# function that checks if graph is fully connected
# scales linearly with number of intermediates
# quick implementation...
def connectivity_checker(edges: chex.Array, info: GraphInfo) -> chex.Array:       
	in_sum = jnp.sum(edges, axis=1)
	out_sum = jnp.sum(edges, axis=0)
	ins_connected = jnp.not_equal(in_sum, 0)[info.num_inputs:]
	outs_connected = jnp.not_equal(out_sum, 0)[:info.num_intermediates]
	return jnp.logical_not(jnp.logical_xor(ins_connected, outs_connected))
        
