import copy
from typing import Tuple
from functools import partial

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import chex

from .core import GraphInfo, vertex_eliminate


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
    t: chex.Numeric
    info: GraphInfo
    edges: chex.Array
    vertices: chex.Array
    attn_mask: chex.Array
    
    def __init__(self,
                t: int, 
                edges: chex.Array,
                vertices: chex.Array,
                attn_mask: chex.Array) -> None:
        self.t = t
        self.edges = edges
        self.vertices = vertices
        self.attn_mask = attn_mask


def gamestate_flatten(vgs: VertexGameState):
    children = (vgs.t, vgs.edges, vgs.vertices, vgs.attn_mask)
    aux_data = None
    return children, aux_data


def gamestate_unflatten(aux_data, children) -> VertexGameState:
    return VertexGameState(*children)


# Registering GraphState as a PyTree node so we can use it with vmap and jit
register_pytree_node(VertexGameState,
                    gamestate_flatten,
                    gamestate_unflatten)


def make_vertex_game_state(edges: chex.Array, 
                           info: GraphInfo,
                           vertices: chex.Array = None,
                           attn_mask: chex.Array = None) -> VertexGameState:
    """TODO add docstring

    Args:
        edges (chex.Array): _description_
        info (GraphInfo): _description_
        vertices (chex.Array, optional): _description_. Defaults to None.
        attn_mask (chex.Array, optional): _description_. Defaults to None.

    Returns:
        VertexGameState: _description_
    """
    vertices = jnp.zeros(info.num_intermediates) if vertices is None else vertices
    mask = jnp.ones((info.num_intermediates, info.num_intermediates))
    attn_mask = mask if attn_mask is None else attn_mask
    return VertexGameState(t=jnp.where(vertices > 0., 1, 0).sum(),  
                            edges=edges, 
                            vertices=vertices,
                            attn_mask=attn_mask)
    

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
    info: GraphInfo
    
    def __init__(self, info: GraphInfo) -> None:
        super().__init__()
        self.info = info
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            vgs: VertexGameState,
            action: int) -> Tuple[VertexGameState, float, bool]:  
        # Actions go from 0 to num_intermediates-1 
        # and vertices go from 1 to num_intermediates      
        vertex = action + 1
        t = vgs.t.astype(jnp.int32)

        edges = vgs.edges
        new_edges, nops = vertex_eliminate(vertex, edges, self.info)
        obs = lax.slice_in_dim(new_edges, 0, self.info.num_intermediates, axis=1)
                
        vgs.t += 1
        vgs.edges = vgs.edges.at[:, :].set(new_edges)
        vgs.vertices = vgs.vertices.at[t].set(vertex)
        vgs.attn_mask = vgs.attn_mask.at[:, action].set(0.)
        vgs.attn_mask = vgs.attn_mask.at[action, :].set(0.)

        # Reward is the negative of the multiplication count
        reward = -nops
        
        terminated = lax.cond((t == self.info.num_intermediates-1).all(),
                                lambda: True,
                                lambda: False)

        return obs, vgs, reward, terminated

