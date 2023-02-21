""" 
Edge and vertex elimination functions for Cross-Country elimination 
that are totally jit-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761
"""
from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex


def front_eliminate(gs_edges: chex.Array, 
                    edge: Tuple[int, int],
                    info: chex.Array) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements the front-elimination procedure
    on the edges of a GraphState object.

    Arguments:
        - gs_edges (chex.Array): Edges contained in a GraphState object that 
                                describes the computational graph where we want 
                                to front-eliminate the given edge.
        - edge (Tuple[int, int]): Tuple of integers describing the edge we want
                                to eliminate.
        - info (chex.Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    num_inputs, _, _, num_edges, _ = info
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs_edges[e0, e1]
    gs_edges = gs_edges.at[e0, e1].set(0.)

    def front_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = gs_edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(i == edge[1], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, e0, nonzeros[1], val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, None
    nonzeros = jnp.stack(jnp.nonzero(gs_edges, 
                                    size=num_edges,
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(front_update_edge, (gs_edges, 0), nonzeros)
    return output
 

def back_eliminate(gs_edges: chex.Array, 
                   edge: Tuple[int, int],
                   info: chex.Array) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements the back-elimination procedure
    on the edges of a GraphState object.

    Arguments:
        - gs_edges (chex.Array): Edges contained in a GraphState object that 
                                describes the computational graph where we want 
                                to front-eliminate the given edge.
        - edge (Tuple[int, int]): Tuple of integers describing the edge we want
                                to eliminate.
        - info (chex.Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    num_inputs, _, _, num_edges, _ = info
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs_edges[e0, e1]
    gs_edges = gs_edges.at[e0, e1].set(0.)

    def back_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = gs_edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(j == edge[0], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, nonzeros[0], e1, val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, None
    
    nonzeros = jnp.stack(jnp.nonzero(gs_edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(back_update_edge, (gs_edges, 0), nonzeros)
    return output


def eliminate(gs_edges: chex.Array, 
            vertex: int, 
            info: chex.Array) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements the vertex-elimination procedure
    on a GraphState object. Vertex elimination means that we front-eliminate
    all incoming edges and back-eliminate all outgoing edges of a given vertex.

    Arguments:
        - gs_edges (GraphState): GraphState that describes the computational graph 
                            where we want to front-eliminate the given edge.
        - vertex (int): Vertex we want to eliminate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    num_inputs, _, _, num_edges, _ = info
    
    def update_edges(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
                        
        edges, fops = lax.cond(j == vertex,
                                lambda x, m, n: front_eliminate(x, (m,n), info), 
                                lambda x, m, n: (x, 0), 
                                edges, i, j)

        edges, bops = lax.cond(i == vertex, 
                                lambda x, m, n: back_eliminate(x, (m,n), info), 
                                lambda x, m, n: (x, 0), 
                                edges, i, j)
        
        
        nops += (fops + bops)
        carry = (edges, nops)
        return carry, None
        
    nonzeros = jnp.stack(jnp.nonzero(gs_edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(update_edges, (gs_edges, 0), nonzeros)
    # gs = output[0]
    # num_steps = gs.info[4]
    # gs.info = gs.info.at[4].add(1)
    # gs.state = gs.state.at[num_steps].set(vertex)
    return output


def forward(gs_edges: chex.Array, info: chex.Array) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements forward-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n.

    Arguments:
        - gs (GraphState): GraphState that describes the computational graph 
                            where we want to differntiate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    _, num_intermediates, _, _, _ = info
    
    def fwd(carry, vertex):
        edges, nops = carry
        edges, ops = eliminate(edges, vertex, info)
        nops += ops
        carry = (edges, nops)
        return carry, None
    vertex_list = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(fwd, (gs_edges, 0), vertex_list)
    return output


def reverse(gs_edges: chex.Array, info: chex.Array) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements reverse-mode AD by 
    eliminating the vertices in sequential order n,n-1,...,2,1.

    Arguments:
        - gs (GraphState): GraphState that describes the computational graph 
                            where we want to differntiate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    _, num_intermediates, _, _, _ = info
    
    def rev(carry, vertex):
        edges, nops = carry
        edges, ops = eliminate(edges, vertex, info)
        nops += ops
        carry = (edges, nops)
        return carry, None
    vertex_list = jnp.arange(1, num_intermediates+1)[::-1]
    output, _ = lax.scan(rev, (gs_edges, 0), vertex_list)
    return output


def add_edge(edges: chex.Array, 
            pos: Tuple[int, int], 
            info: chex.Array) -> chex.Array:
    """TODO refine documentation
    Jittable function to add a new edge to a GraphState object, i.e. a new
    entry to the `edges` matrix.

    Input vertices range from `-num_inputs+1` to 0, while the last `num_output` 
    vertices are the output vertices.

    Arguments:
        - edges (GraphState): GraphState object where we want to add the edge.
        - pos (Tuple[int, int]): Tuple that describes which two vertices are 
                                connected, i.e. pos = (from, to).
        - info (Array): Contains meta data about the computational graph.
    """
    num_inputs, _, _, _, _ = info
    return edges.at[pos[0]+num_inputs-1, pos[1]-1].set(1), info.at[3].add(1)

