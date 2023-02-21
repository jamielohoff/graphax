import copy
from typing import Callable, Tuple
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import chex

from .core import eliminate, front_eliminate, back_eliminate


class VertexGameState:
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
        """
        Returns the graph.info array as a tuple.
        """
        return tuple(int(i) for i in self.info)


def graphstate_flatten(gs: GraphState):
    children = (gs.info, gs.edges, gs.state)
    aux_data = None
    return children, aux_data


def graphstate_unflatten(aux_data, children) -> GraphState:
    return GraphState(*children)


# Registering GraphState as a PyTree node so we can use it with vmap and jit
register_pytree_node(GraphState,
                    graphstate_flatten,
                    graphstate_unflatten)


def is_bipartite(gs: VertexGameState) -> bool:
    """Alternative implementation that makes use of the game state for faster computation
    Jittable function to test if a graph is bipartite by comparing the number of
    non-zero entries in gs.states to the number of intermediate variables, i.e.
    gs.info.at[1].get().

    Arguments:
        - gs (GraphState): GraphState object we want to check.
    """
    return jnp.count_nonzero(gs.state) == gs.info.at[1].get() # num_intermediates


def vert_elim(info: chex.Array, gs: GraphState, vertex: int):
    return eliminate(gs, vertex, info)


def front_elim(info: chex.Array, gs: GraphState, edge: Tuple[int, int]):
    return front_eliminate(gs, edge, info)


def back_elim(info: chex.Array, gs: GraphState, edge: Tuple[int, int]):
    return back_eliminate(gs, edge, info)


class VertexGame:
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
    gs: GraphState
    vertex_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        self.vertex_eliminate = partial(vert_elim, gs.get_info())
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: int) -> Tuple[GraphState, float, bool]:  
        # Actions go from 0 to num_intermediates-1 
        # and vertices go from 1 to num_intermediates      
        vertex = action + 1
        
        new_gs, nops = self.vertex_eliminate(gs, vertex)

        # Reward is the negative of the multiplication count
        reward = -nops
        
        terminated = lax.cond(is_bipartite(new_gs),
                                    lambda: True,
                                    lambda: False)
    
        return new_gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> GraphState:
        return copy.deepcopy(self.gs)


# TODO add comments and docstring
class EdgeGame:
    """
    OpenAI-like gymnax environment for the game
    
    The game always has finite termination range!
    """
    gs: GraphState
    num_inputs: int
    front_eliminate: Callable
    back_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        self.num_inputs = gs.get_info()[0]
        self.num_intermediates = gs.get_info()[1]
        self.num_outputs = gs.get_info()[2]
        self.front_eliminate = partial(front_elim, gs.get_info())
        self.back_eliminate = partial(back_elim, gs.get_info())
    
    # Remove JIT compilation here
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: chex.Array) -> Tuple[GraphState, float, bool]:
        i, j, mode = action

        i = i - self.num_inputs + 1
        j += 1

        new_gs, nops = lax.cond(mode == 0,
                            lambda g: self.front_eliminate(g, (i,j)),
                            lambda g: self.back_eliminate(g, (i,j)),
                            gs)
                
        reward = -nops
    
        terminated = lax.cond(is_bipartite(new_gs),
                            lambda: True,
                            lambda: False)
        return new_gs, reward, terminated
    
    # remove jit compilation here!
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> GraphState:
        return copy.deepcopy(self.gs)

