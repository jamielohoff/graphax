from typing import Tuple, Any

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

from chex import Array, Numeric

from .core import vertex_eliminate, get_info


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
    t: Numeric
    edges: Array
    vertices: Array
    attn_mask: Array
    
    def __init__(self,
                t: Numeric, 
                edges: Array,
                vertices: Array,
                attn_mask: Array) -> None:
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


# Registering VertexGameState as a PyTree node so we can use it with 
# jax.vmap and jax.jit
register_pytree_node(VertexGameState,
                    gamestate_flatten,
                    gamestate_unflatten)


def make_vertex_game_state(edges: Array, 
                           vertices: Array = None,
                           attn_mask: Array = None) -> VertexGameState:
    """TODO add docstring

    Args:
        edges (chex.Array): _description_
        info (GraphInfo): _description_
        vertices (chex.Array, optional): _description_. Defaults to None.
        attn_mask (chex.Array, optional): _description_. Defaults to None.

    Returns:
        VertexGameState: _description_
    """
    num_i, num_v = get_info(edges)
    vertices = jnp.zeros(num_v) if vertices is None else vertices
    mask = jnp.ones((num_i, num_i))
    attn_mask = mask if attn_mask is None else attn_mask
    return VertexGameState(t=jnp.where(vertices > 0, 1, 0).sum(),  
                            edges=edges, 
                            vertices=vertices,
                            attn_mask=attn_mask)
    

EnvOut = Tuple[VertexGameState, float, bool, Any]

    
def step(vgs: VertexGameState, action: int) -> EnvOut:  
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
    t = vgs.t.astype(jnp.int32)
    num_i, num_v = get_info(vgs.edges)

    edges = vgs.edges
    new_edges, nops = vertex_eliminate(vertex, edges)
    obs = lax.slice_in_dim(new_edges, 1, -1, axis=1).astype(jnp.float32)
            
    vgs.t += 1
    vgs.edges = vgs.edges.at[:, :].set(new_edges)
    vgs.vertices = vgs.vertices.at[t].set(vertex)
    vgs.attn_mask = vgs.attn_mask.at[:, action].set(0)
    vgs.attn_mask = vgs.attn_mask.at[action, :].set(0)

    # Reward is the negative of the multiplication count
    reward = -nops
    terminated = lax.cond((t == num_v-1).all(), lambda: True, lambda: False)

    return obs, vgs, reward, terminated

