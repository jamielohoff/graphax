from typing import Tuple, Any

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from .core import (vertex_eliminate, 
                    get_elimination_order, 
                    get_vertex_mask, 
                    get_shape)
    

EnvOut = Tuple[Array, float, bool, Any]
    
def step(edges: Array, action: int) -> EnvOut:  
    """
    OpenAI-like environment for a game where to goal is to find the 
    best vertex elimination order with minimal multiplication count.
    This game always has finite termination range.

    The `state` of the game is essentially the matrix containing the edges of the
    computational graph and the array containing the edges that have already been
    eliminated.

    The `reward` is the negative number of multiplications since we want to 
    minimize that.

    The `action space` is equal to the number of remaining vertices that can be
    eliminated. For example, for 10 intermediate variables, there are 10 
    different actions. However, every action can only be executed once. This is
    realized by action masking where the logits of actions that have already 
    been performed are sent to -inf.

    The `termination` of the game is indicated by the is_bipartite feature, i.e.
    the game is over when all intermediate vertices and edges have been eliminated.
    """
    # Actions go from 0 to num_intermediates-1 
    # and vertices go from 1 to num_intermediates      
    vertex = action + 1
    t = jnp.where(get_elimination_order(edges) > 0, 1, 0).sum()
    new_edges, nops = vertex_eliminate(vertex, edges)
    new_edges = new_edges.at[3, 0, t].set(vertex)
    
    # Reward is the negative of the multiplication count
    reward = -nops
    num_eliminated_vertices = get_vertex_mask(new_edges).sum()
    num_intermediates = get_shape(new_edges)[1]
    terminated = lax.cond(num_eliminated_vertices == num_intermediates, lambda: True, lambda: False)

    return new_edges, reward, terminated

