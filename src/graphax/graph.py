from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import chex


class GraphState:
    """The state of the graph is the connectivity of the edges represented
    as a sparse matrix plus the already eliminated vertices

    The `info` field contains meta information about the computational graph in
    the following order:
        1.) num_inputs
        2.) num_intermediates
        3.) num_outputs
        4.) num_edges
        5.) num_steps = number of vertices that have been eliminated
    Args:
        NamedTuple (_type_): _description_
    """
    info: chex.Array
    edges: chex.Array
    state: chex.Array
    
    def __init__(self, 
                info: chex.Array,
                edges: chex.Array,
                state: chex.Array) -> None:
        self.info = info
        self.edges = edges
        self.state = state
        
    def get_info(self) -> Tuple[int, int, int, int, int]:
        return tuple(int(i) for i in self.info)


def graphstate_flatten(gs: GraphState):
    children = (gs.info, gs.edges, gs.state)
    aux_data = None
    return children, aux_data


def graphstate_unflatten(aux_data, children) -> GraphState:
    return GraphState(*children)


register_pytree_node(GraphState,
                    graphstate_flatten,
                    graphstate_unflatten)


def add_edge(gs: GraphState, 
            pos: Tuple[int, int], 
            val: float, 
            info: chex.Array) -> GraphState:
    """
    TODO add docstring
    """
    num_inputs, _, _, _, _ = info
    gs.edges = gs.edges.at[pos[0]+num_inputs-1, pos[1]-1].set(val)
    gs.info = gs.info.at[3].add(1)
    return gs


def is_bipartite(gs: GraphState) -> bool:
    """
    TODO docstring

    Returns:
        _type_: _description_
    """
    return jnp.count_nonzero(gs.state) == gs.info.at[1].get() # num_intermediates

