""" 
GPU-friendly edge and vertex elimination procedures for Cross-Country Elimination 
that are totally JIT-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761

DO NOT TOUCH!
"""
from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array


Edge = Tuple[int, int]


class GraphInfo(NamedTuple):
    """
    Meta-information about the computational graph.
    """
    num_inputs: int
    num_intermediates: int
    num_outputs: int
    num_edges: int


def make_empty_edges(info: GraphInfo) -> Array:
    """
    Creates an empty matrix fo represent the connectivity of the computational graph.
    """
    num_i = info.num_inputs
    num_v = info.num_intermediates
    return jnp.zeros((num_i+num_v, num_v))


def make_graph_info(info: Array) -> GraphInfo:
    """
    Create GraphInfo object from input numpy array or list.
    """
    num_i = info[0]
    num_v = info[1]
    num_edges = (num_i+num_v)*(num_v) - num_v*(num_v-1)//2
    num_edges = int(.5*num_edges)
    return GraphInfo(num_inputs=info[0],
                    num_intermediates=info[1],
                    num_outputs=info[2],
                    num_edges=num_edges)
    
    
def front_eliminate(edge: Edge, edges: Array, info: GraphInfo) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the front-elimination procedure
    on the edge representation of a computational graph.

    Arguments:
        edge (Edge): Edge we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphInfo): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops). 
    """
    num_inputs = info.num_inputs
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] + num_inputs - 1
    
    edges = edges.at[e0, e1-num_inputs].set(0.)
    row = edges.at[e0, :].get()
    _row = edges.at[e1, :].get()
    fmas = jnp.sum(_row)
    
    new_row = jnp.where(_row > 0., _row, row)
    edges = edges.at[e0, :].set(new_row)     
    
    # Cleanup of unneeded edges
    edges = lax.cond(jnp.sum(edges.at[:, e1-num_inputs].get()) == 0.,
                    lambda e: e.at[e1, :].set(0.),
                    lambda e: e,
                    edges)
    
    return edges, fmas
 
 
def back_eliminate(edge: Edge, edges: Array, info: GraphInfo) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the back-elimination procedure
    on the edges of a GraphState object.

    Arguments:
        edge (Edge): Edge we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphState): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_inputs = info.num_inputs
    
    e0 = edge[0] - 1
    e1 = edge[1] - 1
    
    edges = edges.at[e0+num_inputs, e1].set(0.)
    col = edges.at[:, e1].get()
    _col = edges.at[:, e0].get()
    fmas = jnp.sum(_col)
    
    new_col = jnp.where(_col > 0., _col, col)
    edges = edges.at[:, e1].set(new_col)   
    
    # Cleanup of unneeded edges
    edges = lax.cond(jnp.sum(edges.at[e0+num_inputs, :].get()) == 0.,
                    lambda e: e.at[:, e0].set(0.),
                    lambda e: e,
                    edges)
    
    return edges, fmas


def vertex_eliminate(vertex: int, edges: Array, info: GraphInfo) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the vertex-elimination procedure. 
    Vertex elimination means that we front-eliminate all incoming edges and 
    back-eliminate all outgoing edges of a given vertex. However, the implementation
    here does not make use of the function above to be more efficient.

    Arguments:
        vertex (int): Vertex we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphInfo): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_inputs = info.num_inputs
    num_intermediates = info.num_intermediates

    col = edges.at[:, vertex-1].get()
    _fmas = col.sum()
    
    def update_edges_fn(carry, nonzero):
        _edges, fmas = carry

        _col = _edges.at[:, nonzero].get()
        new_col = jnp.where(_col > 0., _col, col)
        _edges = _edges.at[:, nonzero].set(new_col)     
        
        fmas = lax.cond(nonzero > -1, lambda x: x+_fmas, lambda x: x, fmas)
        carry = (_edges, fmas)
        return carry, None
        
    nonzeros = jnp.nonzero(edges.at[num_inputs+vertex-1, :].get(),
                           size=num_intermediates,
                           fill_value=-1)[0].T
    output, _ = lax.scan(update_edges_fn, (edges, 0.), nonzeros)
    edges, fmas = output
    edges = edges.at[num_inputs+vertex-1, :].set(0)
    edges = edges.at[:, vertex-1].set(0)
    return edges, fmas


def forward(edges: Array, info: GraphInfo, vertex_mask: Array) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements forward-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphInfo): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_intermediates = info.num_intermediates
    
    def fwd_fn(carry, vertex):
        _edges, fmas = carry
        is_masked = jnp.any((vertex == vertex_mask))
        _edges, _fmas = lax.cond(is_masked,
                                lambda e: (e, 0.),
                                lambda e: vertex_eliminate(vertex, e, info),
                               _edges)
        fmas += _fmas
        carry = (_edges, fmas)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(fwd_fn, (edges, 0.), vertices)
    return output


def reverse(edges: Array, info: GraphInfo, vertex_mask: Array) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements reverse-mode AD by 
    eliminating the vertices in sequential order n,n-1,...,2,1 and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphInfo): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_intermediates = info.num_intermediates
    
    def rev_fn(carry, vertex):
        _edges, fmas = carry
        is_masked = jnp.any((vertex == vertex_mask))
        _edges, _fmas = lax.cond(is_masked,
                                lambda e: (e, 0.),
                                lambda e: vertex_eliminate(vertex, e, info),
                               _edges)
        fmas += _fmas
        carry = (_edges, fmas)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)[::-1]
    output, _ = lax.scan(rev_fn, (edges, 0.), vertices)
    return output

