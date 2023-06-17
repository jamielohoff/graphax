""" 
Edge and vertex elimination functions for Cross-Country elimination 
that are totally jit-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761

DO NOT TOUCH!
"""
from typing import NamedTuple, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

import chex


class GraphInfo(NamedTuple):
    """
    Meta-information about the computational graph
    """
    num_inputs: int
    num_intermediates: int
    num_outputs: int
    num_edges: int


def make_empty_edges(info: GraphInfo) -> chex.Array:
    num_i = info.num_inputs
    num_v = info.num_intermediates
    return jnp.zeros((num_i+num_v, num_v))


def make_graph_info(info: chex.Array) -> GraphInfo:
    num_i = info[0]
    num_v = info[1]
    num_edges = (num_i+num_v)*(num_v) - num_v*(num_v-1)//2
    num_edges = int(.5*num_edges)
    return GraphInfo(num_inputs=info[0],
                    num_intermediates=info[1],
                    num_outputs=info[2],
                    num_edges=num_edges)
    
    
def front_eliminate(edges: chex.Array, 
                    edge: Tuple[int, int],
                    info: GraphInfo) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements the front-elimination procedure
    on the edges of a GraphState object.

    Arguments:
        - edges (chex.Array): Edges contained in a GraphState object that 
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
    num_inputs = info.num_inputs
    num_edges = info.num_edges
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = edges[e0, e1]
    edges = edges.at[e0, e1].set(0.)

    def front_update_edge(carry, nonzeros):
        _edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = edges.at[nonzeros[0], nonzeros[1]].get()
        _edges, ops = lax.cond(i == edge[1], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            _edges, e0, nonzeros[1], val*edge_val)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    nonzeros = jnp.stack(jnp.nonzero(edges, 
                                    size=num_edges,
                                    fill_value=-num_edges)).T
    output, _ = lax.scan(front_update_edge, (edges, 0), nonzeros)
    return output
 

def back_eliminate(edges: chex.Array, 
                   edge: Tuple[int, int],
                   info: GraphInfo) -> Tuple[chex.Array, int]:
    """TODO fix docstring
    Fully jit-compilable function that implements the back-elimination procedure
    on the edges of a GraphState object.

    Arguments:
        - edges (chex.Array): Edges contained in a GraphState object that 
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
    num_inputs = info.num_inputs
    num_edges = info.num_edges
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = edges[e0, e1]
    edges = edges.at[e0, e1].set(0.)

    def back_update_edge(carry, nonzeros):
        _edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = edges.at[nonzeros[0], nonzeros[1]].get()
        _edges, ops = lax.cond(j == edge[0], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            _edges, nonzeros[0], e1, val*edge_val)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    
    nonzeros = jnp.stack(jnp.nonzero(edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).T
    output, _ = lax.scan(back_update_edge, (edges, 0), nonzeros)
    return output


def vertex_eliminate(edges: chex.Array, 
                    vertex: int, 
                    info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_inputs = info.num_inputs
    num_edges = info.num_edges
    
    def update_edges(carry, nonzeros):
        _edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
                        
        _edges, fops = lax.cond(j == vertex,
                                lambda x, m, n: front_eliminate(x, (m, n), info), 
                                lambda x, m, n: (x, 0), 
                                _edges, i, j)

        _edges, bops = lax.cond(i == vertex, 
                                lambda x, m, n: back_eliminate(x, (m, n), info), 
                                lambda x, m, n: (x, 0), 
                                _edges, i, j)
        
        nops += (fops + bops)
        carry = (_edges, nops)
        return carry, None
        
    nonzeros = jnp.stack(jnp.nonzero(edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).T
    output, _ = lax.scan(update_edges, (edges, 0), nonzeros)
    return output


def vertex_eliminate_gpu(vertex: int, 
                        edges: chex.Array,
                        info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_inputs = info.num_inputs
    num_intermediates = info.num_intermediates
    num_outputs = info.num_outputs
    num_edges = info.num_edges

    col = edges.at[:, vertex-1].get()
    ops = col.sum()
    
    def update_edges(carry, nonzero):
        _edges, nops = carry

        _col = _edges.at[:, nonzero].get()
        new_col = jnp.where(_col > 0., _col, col)
        _edges = _edges.at[:, nonzero].set(new_col, mode="drop")     
        
        nops = lax.cond(nonzero > -1, lambda x: x+ops, lambda x: x, nops)
        carry = (_edges, nops)
        return carry, None
        
    nonzeros = jnp.nonzero(edges.at[num_inputs+vertex-1, :].get(), 
                                    size=num_intermediates+num_outputs, 
                                    fill_value=-num_edges)[0]
    output, _ = lax.scan(update_edges, (edges, 0.), nonzeros)
    edges, nops = output
    edges = edges.at[num_inputs+vertex-1, :].set(0)
    edges = edges.at[:, vertex-1].set(0)
    return edges, nops


def forward(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_intermediates = info.num_intermediates
    
    def fwd(carry, vertex):
        _edges, nops = carry
        _edges, ops = vertex_eliminate(_edges, vertex, info)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(fwd, (edges, 0.), vertices)
    return output


def forward_gpu(edges: chex.Array, vertex_mask: chex.Array, info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_intermediates = info.num_intermediates
    
    def fwd(carry, vertex):
        _edges, nops = carry
        is_masked = jnp.bool_((vertex == vertex_mask).sum())
        _edges, ops = lax.cond(is_masked,
                                lambda e: (e, 0.),
                                lambda e: vertex_eliminate_gpu(vertex, e, info),
                               _edges)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(fwd, (edges, 0.), vertices)
    return output


def reverse(edges: chex.Array, info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_intermediates = info.num_intermediates
    
    def rev(carry, vertex):
        _edges, nops = carry
        _edges, ops = vertex_eliminate(_edges, vertex, info)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)[::-1]
    output, _ = lax.scan(rev, (edges, 0.), vertices)
    return output


def reverse_gpu(edges: chex.Array, vertex_mask: chex.Array, info: GraphInfo) -> Tuple[chex.Array, int]:
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
    num_intermediates = info.num_intermediates
    
    def rev(carry, vertex):
        _edges, nops = carry
        is_masked = jnp.bool_((vertex == vertex_mask).sum())
        _edges, ops = lax.cond(is_masked,
                                lambda e: (e, 0.),
                                lambda e: vertex_eliminate_gpu(vertex, e, info),
                               _edges)
        nops += ops
        carry = (_edges, nops)
        return carry, None
    vertices = jnp.arange(1, num_intermediates+1)[::-1]
    output, _ = lax.scan(rev, (edges, 0.), vertices)
    return output

