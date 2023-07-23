""" 
GPU-friendly edge and vertex elimination procedures for Cross-Country Elimination 
that are totally JIT-compilable. For an in-depth discussion of Cross-Country 
Elimination and the methods described here see the book 
`Evaluating Derivatives` by Griewank et al., 2008,
https://doi.org/10.1137/1.9780898717761

DO NOT TOUCH!
"""

from functools import partial
from typing import Sequence, Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

# Every entry in the 3-dimensional tensor has the following meaning:
# (sparsity type, Jacobian shape 1st input component == 1st component, 
#                 Jacobain shape 2nd input component == 2nd component,
#                 Jacobian shape 1st output component == 3rd component,
#                 Jacobian shape 2nd output component == 4th component)
# Thus the current implementation can only deal with scalars vectors and matrices
# and related operations. 
# NOTE: No support for higher-order tensors!

# Sparsity types explanation:
# 0: No edge between vertices
# 1: Dense Jacobian, i.e. no Kronecker symbols
# 8: For "copy" operation that keep sparsity
#
# Kronecker symbol between components: 
# 2: (1, 3)
# 3: (2, 4)
# 4: (1, 4)
# 5: (2, 3)
# 6: (1, 3) and (2, 4)
# 7: (1, 4) and (2, 3)

# Row idx is incoming edge, col idx is outgoing edge
# For meaning of the different numbers, "checkout fmas_sparsity_map()" function
CONTRACTION_MAP =  jnp.array([[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 1, 0, 1, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              
                              [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 1, 1, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0]]])    

# Row idx is incoming edge, col idx is outgoing edge
# Gives the resulting sparsity type if two hyperdimensional Jacobians
# are multiplied with each other
MUL_SPARSITY_MAP = jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1],
                              [0, 1, 2, 1, 4, 1, 2, 4, 2],
                              [0, 1, 1, 3, 1, 5, 3, 5, 3],
                              [0, 1, 1, 4, 1, 2, 4, 2, 4],
                              [0, 1, 5, 1, 3, 1, 5, 3, 5],
                              [0, 1, 2, 3, 4, 5, 6, 7, 6],
                              [0, 1, 5, 4, 3, 2, 7, 6, 7],
                              [0, 1, 2, 3, 4, 5, 6, 7, 8]])

# Row idx is incoming edge, col idx is outgoing edge
# Gives the resulting sparsity type if two hyperdimensional Jacobians
# are added to each other
ADD_SPARSITY_MAP = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 0],
                              [1, 1, 1, 1, 1, 1, 1, 1, 1],
                              [2, 1, 2, 1, 1, 1, 2, 1, 2],
                              [3, 1, 1, 3, 1, 1, 3, 1, 3],
                              [4, 1, 1, 1, 4, 1, 1, 4, 4],
                              [5, 1, 1, 1, 1, 5, 1, 5, 5],
                              [6, 1, 2, 3, 1, 1, 6, 1, 6],
                              [7, 1, 1, 1, 4, 5, 1, 7, 7],
                              [0, 1, 2, 3, 4, 5, 6, 7, 8]])


Edge = Tuple[int, int]


def get_info(edges: Array):
    num_v = edges.shape[2]
    num_i = edges.shape[1] - num_v - 1
    return num_i, num_v


def make_empty_edges(info: Array) -> Array:
    """
    Creates an empty matrix fo represent the connectivity of the computational graph.
    """
    num_i = info[0]
    num_v = info[1]
    return jnp.zeros((5, num_i+num_v+1, num_v), dtype=jnp.int32)


@partial(jax.vmap, in_axes=(0, 0))
def sparsity_where(in_edge, out_edge):
    # takes care of the corner cases where there already exists an edge with a 
    # different sparsity type
    i = in_edge.astype(jnp.int32)
    j = out_edge.astype(jnp.int32)
    return ADD_SPARSITY_MAP[i, j]


@partial(jax.vmap, in_axes=(1, None))
def sparsity_fmas_map(in_edge, out_edge):
    """
    TODO add documentation here!
    """
    i = in_edge[0].astype(jnp.int32)
    j = out_edge[0].astype(jnp.int32)
    new_sparsity_type = MUL_SPARSITY_MAP[i, j]
    contraction_map = CONTRACTION_MAP[:, i, j]
    
    masked_factors = lax.cond(jnp.sum(contraction_map) > 0,
                                lambda a: jnp.where(contraction_map > 0, a, 1),
                                lambda a: jnp.zeros(4, dtype=jnp.int32),
                                out_edge[1:])

    fmas = jnp.prod(in_edge[1:3])*jnp.prod(masked_factors)
    return new_sparsity_type, fmas


def get_fmas_jacprod(_jac_edges, fmas, col, _col, nonzero, vertex, num_i):
    # Define aliases
    col_ins = col.at[1:3, :].get()
    col_outs = col.at[3:, :].get()
    
    _col_ins = _col.at[1:3, :].get()
    _col_outs = _col.at[3:, :].get()
        
    # Calculate fmas
    new_sparsity_col, _fmas = sparsity_fmas_map(col, _col[:, vertex+num_i-1])
    new_sparsity_col = sparsity_where(_col[0, :], new_sparsity_col)
    new_sparsity_col = jnp.broadcast_to(new_sparsity_col, (1, *new_sparsity_col.shape))
    fmas = jnp.sum(_fmas)
    # In shape column
    new_col_ins = jnp.where(col_ins[1] > 0, col_ins, _col_ins)
    
    # Out shape column
    new_col_outs = jnp.broadcast_to(_col_outs[:, vertex+num_i-1, jnp.newaxis], _col_outs.shape)
    new_col_outs = jnp.where(col_outs[1] > 0, new_col_outs, _col_outs)
    new_col = jnp.concatenate((new_sparsity_col, new_col_ins, new_col_outs), axis=0)
        
    # Set the Jacobian adjacency matrix
    _jac_edges = lax.dynamic_update_index_in_dim(_jac_edges, new_col, nonzero, -1)
            
    return _jac_edges, fmas


def vertex_eliminate(vertex: int, edges: Array) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the vertex-elimination procedure. 
    Vertex elimination means that we front-eliminate all incoming edges and 
    back-eliminate all outgoing edges of a given vertex. However, the implementation
    here does not make use of the function above to be more efficient.

    Arguments:
        vertex (int): Vertex we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_v = get_info(edges)
    jac_edges = edges.at[:, 1:, :].get()
    col = jac_edges.at[:, :, vertex-1].get()
        
    def update_edges_fn(carry, nonzero):
        _jac_edges, fmas = carry
        # Get the index of the operation and the 
        _col = _jac_edges.at[:, :, nonzero].get()
        # Calculate the fma operations and the new shapes of the Jacobians for 
        # the respective and update the vertex
        _jac_edges, _fmas = lax.cond(nonzero > -1, 
                                    lambda _e, f, c, _c, nz, v: get_fmas_jacprod(_e, f, c, _c, nz, v, num_i), 
                                    lambda _e, f, c, _c, nz, v: (_e, 0), 
                                    _jac_edges, fmas, col, _col, nonzero, vertex)
        fmas += _fmas        
        carry = (_jac_edges, fmas)
        return carry, None
    
    nonzeros = jnp.nonzero(jac_edges.at[0, num_i+vertex-1, :].get(),
                           size=num_v,
                           fill_value=-1)[0].T
        
    output, _ = lax.scan(update_edges_fn, (jac_edges, 0), nonzeros)
    jac_edges, fmas = output
    jac_edges = jac_edges.at[:, num_i+vertex-1, :].set(0)
    jac_edges = jac_edges.at[:, :, vertex-1].set(0)

    edges = edges.at[1, 0, vertex-1].set(1)
    edges = edges.at[:, 1:, :].set(jac_edges)
    return edges, fmas


def cross_country(order: Sequence[int], edges: Array) -> Array:
    """
    Fully JIT-compilable function that implements cross-country elimination 
    according to the given order.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_v = get_info(edges)
    vertex_mask = 1 + jnp.nonzero(1 - edges.at[1, 0, :].get(), size=num_v, fill_value=-2)[0]
    def cc_fn(carry, vertex):
        _edges, fmas = carry
        not_masked = jnp.any(vertex == vertex_mask)
        _edges, _fmas = lax.cond(not_masked,
                                lambda e: vertex_eliminate(vertex, e),
                                lambda e: (e, 0),
                               _edges)
        fmas += _fmas
        carry = (_edges, fmas)
        return carry, None
    vertices = jnp.array(order)
    output, _ = lax.scan(cc_fn, (edges, 0), vertices)
    return output


def forward(edges: Array):
    """
    Fully JIT-compilable function that implements forward-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_v = get_info(edges)
    order = jnp.arange(1, num_v+1)
    output = cross_country(order, edges)
    return output


def reverse(edges: Array):
    """
    Fully JIT-compilable function that implements reverse-mode AD by 
    eliminating the vertices in sequential order 1,2,3,...,n-1,n and ignores
    the ones that are given by vertex_mask, because these are typically the 
    output vertices.

    Arguments:
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_i, num_v = get_info(edges)
    order = jnp.arange(1, num_v+1)[::-1]
    output = cross_country(order, edges)
    return output

