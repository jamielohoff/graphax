from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node

import chex


class GraphState:
    """
    The state of the graph is the connectivity of the edges represented
    as a matrix and an array tracking which vertices have already have 
    been eliminated.

    The `info` field contains meta information about the computational graph in
    the following order:
<<<<<<< HEAD
        1.) num_inputs
        2.) num_intermediates
        3.) num_outputs
        4.) num_edges
        5.) num_steps = number of vertices that have been eliminated
    Args:
        NamedTuple (_type_): _description_
=======
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
>>>>>>> 9cf7b4408581b74936ac0e26ce669956fb398546
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


def add_edge(gs: GraphState, 
            pos: Tuple[int, int], 
            val: float, 
            info: chex.Array) -> GraphState:
    """
    Jittable function to add a new edge to a GraphState object, i.e. a new
    entry to the `edge` matrix.

    Input vertices range from `-ninputs+1` to 0, while the last `noutput` vertices
    are the output vertices.

    Arguments:
        - gs (GraphState): GraphState object where we want to add the edge.
        - pos (Tuple[int, int]): Tuple that describes which two vertices are 
                                connected, i.e. pos = (from, to).
        - val (float): Which value to assign to the corresponding edge.
        - info (Array): Contains meta data about the computational graph.
    """
    num_inputs, _, _, _, _ = info
    gs.edges = gs.edges.at[pos[0]+num_inputs-1, pos[1]-1].set(val)
    gs.info = gs.info.at[3].add(1)
    return gs


def is_bipartite(gs: GraphState) -> bool:
    """
    Jittable function to test if a graph is bipartite by comparing the number of
    non-zero entries in gs.states to the number of intermediate variables, i.e.
    gs.info.at[1].get().

    Arguments:
        - gs (GraphState): GraphState object we want to check.
    """
    return jnp.count_nonzero(gs.state) == gs.info.at[1].get() # num_intermediates

