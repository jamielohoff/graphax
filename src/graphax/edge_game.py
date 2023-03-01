import copy
from typing import Tuple
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import chex

from .core import GraphInfo, front_eliminate, back_eliminate


class EdgeGameState:
    """TODO refurbish documentation
    The state of the graph is the connectivity of the edges represented
    as a matrix and an array tracking which vertices have already have 
    been eliminated.

    The `info` field contains meta information about the computational graph in
    the following order:
        [0] = number of input variables
        [1] = number of intermediate variables
        [2] = number of output variables
        [3] = number of edges
        [4] = number of vertices that have been eliminated

    The `edge` field contains the connectivity of the computational graph in 
    binary encoding. Especially, if edges[i, j] != 0 then we have an edge going
    from vertex i to vertex j, i.e. i-->j. Since input vertices cannot be
    connected to other input vertices and output vertices cannot be connected
    to any other vertices, the matrix has the shape 
    (ninputs + nintermediates) x (nintermediates + noutputs).
    Thus i runs over the input and intermediate vertices and j runs over the 
    intermediate and output vertices.

    The `state` field contains the elimination order of the vertices in the graph.
    It is only utilized for vertex elimination. The elimination order is 
    indicated by the value of the array at the index equal to the vertex number,
    i.e. if vertex 6 was the 4th index to be eliminated, state[6] = 4.
    Such that for the elimination order 4,2,1,3 we have state = [3, 2, 4, 1, 0].
    Zeros indicate uneliminated vertices, e.g. 5 in the last example.

    Arguments:
        - info (Array): Info contains meta-information about the computational 
                        graph, like the number of input variables, edges etc.
        - edges (Array): A binary matrix containing the connectivity of the 
                        vertices
        - state (Array): An array containing the edges which have already been 
                        eliminated.
    """
    info: GraphInfo
    edges: chex.Array
    
    def __init__(self,
                info: GraphInfo,
                edges: chex.Array) -> None:
        self.info = info
        self.edges = edges


def gamestate_flatten(egs: EdgeGameState):
    children = (egs.info, egs.edges)
    aux_data = None
    return children, aux_data


def gamestate_unflatten(aux_data, children) -> EdgeGameState:
    return EdgeGameState(*children)


# Registering GraphState as a PyTree node so we can use it with vmap and jit
register_pytree_node(EdgeGameState,
                    gamestate_flatten,
                    gamestate_unflatten)


def make_edge_game_state(info: GraphInfo, edges: chex.Array) -> EdgeGameState:
    return EdgeGameState(t=0, info=info, edges=edges)


def is_bipartite(egs: EdgeGameState) -> bool:
    """Alternative implementation that makes use of the game state for faster computation
    jittable function to test if a graph is bipartite by comparing the number of
    non-zero entries in gs.states to the number of intermediate variables, i.e.
    gs.info.at[1].get().

    Arguments:
        - vgs (GraphState): GraphState object we want to check.
    """
    num_i = egs.info.num_inputs
    num_v = egs.info.num_intermediates
    return jnp.sum(egs.edges.at[:num_i, :num_v]) == 0


class EdgeGame:
    """
    OpenAI-like gymnax environment for a game where to goal is to find the 
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
    egs: EdgeGameState
    
    def __init__(self, egs: EdgeGameState) -> None:
        super().__init__()
        self.egs = copy.deepcopy(egs)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            egs: EdgeGameState,
            action: int) -> Tuple[EdgeGameState, float, bool]:  
        # Actions go from 0 to num_intermediates-1 
        # and vertices go from 1 to num_intermediates      
        vertex = action + 1
        t = egs.t.astype(jnp.int32)

        edges = egs.edges
        new_edges, nops = vertex_eliminate(edges, vertex, self.egs.info)

        egs.edges = egs.edges.at[:, :].set(new_edges)

        # Reward is the negative of the multiplication count
        reward = -nops
        
        terminated = lax.cond((t == self.vgs.info.num_intermediates-1).all(),
                            lambda: True,
                            lambda: False)
    
        return egs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey = None) -> EdgeGameState:
        return copy.deepcopy(self.egs)

