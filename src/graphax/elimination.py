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

from jaxtyping import Array

from .graph import GraphState


def front_eliminate(gs: GraphState, 
                    edge: Tuple[int, int],
                    info: Array) -> Tuple[GraphState, int]:
    """
    Fully jit-compilable function that implements the front-elimination procedure
    on a GraphState object.

    Arguments:
        - gs (GraphState): GraphState that describes the computational graph 
                            where we want to front-eliminate the given edge.
        - edge (Tuple[int, int]): Tuple of integers describing the edge we want
                                to eliminate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    ninputs, nintermediates, noutputs, nedges, nsteps = info
    
    e0 = edge[0] + ninputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs.edges[e0, e1]
    gs.edges = gs.edges.at[e0, e1].set(0.)

    def front_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - ninputs + 1
        j = nonzeros[1] + 1
        val = gs.edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(i == edge[1], 
                            lambda x, m, n, val: (x.at[m, n].add(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, e0, nonzeros[1], val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, edges
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=nedges,
                                    fill_value=-nedges)).transpose(1, 0)
    output, _ = lax.scan(front_update_edge, (gs.edges, 0), nonzeros)
    gs.edges = output[0]
    return gs, output[1]
 

def back_eliminate(gs: GraphState, 
                   edge: Tuple[int, int],
                   info: Array) -> Tuple[GraphState, int]:
    """
    Fully jit-compilable function that implements the back-elimination procedure
    on a GraphState object.

    Arguments:
        - gs (GraphState): GraphState that describes the computational graph 
                            where we want to back-eliminate the given edge.
        - edge (Tuple[int, int]): Tuple of integers describing the edge we want
                                to eliminate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    ninputs, nintermediates, noutputs, nedges, nsteps = info
    
    e0 = edge[0] + ninputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs.edges[e0, e1]
    gs.edges = gs.edges.at[e0, e1].set(0.)

    def back_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - ninputs + 1
        j = nonzeros[1] + 1
        val = gs.edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(j == edge[0], 
                            lambda x, m, n, val: (x.at[m, n].add(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, nonzeros[0], e1, val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, edges
    
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=nedges, 
                                    fill_value=-nedges)).transpose(1, 0)
    output, _ = lax.scan(back_update_edge, (gs.edges, 0), nonzeros)
    gs.edges = output[0]
    return gs, output[1]


def eliminate(gs: GraphState, 
            vertex: int, 
            info: Array) -> Tuple[GraphState, int]:
    """
    Fully jit-compilable function that implements the vertex-elimination procedure
    on a GraphState object. Vertex elimination means that we front-eliminate
    all incoming edges and back-eliminate all outgoing edges of a given vertex.

    Arguments:
        - gs (GraphState): GraphState that describes the computational graph 
                            where we want to front-eliminate the given edge.
        - vertex (int): Vertex we want to eliminate.
        - info (Array): Meta-information about the computational graph.

    Returns:
        A tuple that contains a new GraphState object with updated edges and 
        an integer containing the number of multiplications necessary to 
        eliminate the given edge. 
    """
    ninputs, nintermediates, noutputs, nedges, nsteps = info
    
    def update_edges(carry, nonzeros):
        gs, nops = carry
        i = nonzeros[0] - ninputs + 1
        j = nonzeros[1] + 1
                        
        gs, fops = lax.cond(j == vertex,
                            lambda x, i, j: front_eliminate(x, (i,j), info), 
                            lambda x, i, j: (x, 0), 
                            gs, i, j)

        gs, bops = lax.cond(i == vertex, 
                            lambda x, i, j: back_eliminate(x, (i,j), info), 
                            lambda x, i, j: (x, 0), 
                            gs, i, j)
        
        
        nops += (fops + bops)
        carry = (gs, nops)
        return carry, gs
        
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=nedges, 
                                    fill_value=-nedges)).transpose(1, 0)
    output, _ = lax.scan(update_edges, (gs, 0), nonzeros)
    gs = output[0]
    num_steps = gs.info[4]
    gs.info = gs.info.at[4].add(1)
    gs.state = gs.state.at[num_steps].set(vertex)
    return gs, output[1]


def forward(gs: GraphState, info: Array) -> Tuple[GraphState, int]:
    """
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
    ninputs, nintermediates, noutputs, nedges, nsteps = info
    
    def fwd(carry, edge):
        gs, nops = carry
        gs, ops = eliminate(gs, edge, info)
        nops += ops
        carry = (gs, nops)
        return carry, edge
    vertex_list = jnp.arange(1, nintermediates+1)
    output, _ = lax.scan(fwd, (gs, 0), vertex_list)
    gs = output[0]
    return gs, output[1]


def reverse(gs: GraphState, info: Array) -> Tuple[GraphState, int]:
    """
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
    ninputs, nintermediates, noutputs, nedges, nsteps = info
    
    def rev(carry, edge):
        gs, nops = carry
        gs, ops = eliminate(gs, edge, info)
        nops += ops
        carry = (gs, nops)
        return carry, edge
    vertex_list = jnp.arange(1, nintermediates+1)[::-1]
    output, _ = lax.scan(rev, (gs, 0), vertex_list)
    gs = output[0]
    return gs, output[1]

