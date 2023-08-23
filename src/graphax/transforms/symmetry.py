from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from ..core import get_shape

# TODO needs reworking

def swap_rows(i: int, j: int, edges: Array) -> Array:
    val1 = edges.at[i, :].get()
    val2 = edges.at[j, :].get()
    edges = edges.at[i, :].set(val2)
    edges = edges.at[j, :].set(val1)
    return edges


def swap_cols(i: int, j: int, edges: Array) -> Array:
    val1 = edges.at[:, i].get()
    val2 = edges.at[:, j].get()
    edges = edges.at[:, i].set(val2)
    edges = edges.at[:, j].set(val1)
    return edges


def swap_inputs(i: int, j: int, edges: Array) -> Array:
    num_i, num_vo = get_shape(edges)
    return swap_rows(i+num_i-1, j+num_i-1, edges)


def swap_outputs(i: int, j: int, edges: Array) -> Array:
    return swap_cols(edges, i-1, j-1)


def _swap_intermediates(i: int, j: int, edges: Array) -> Array:
    num_i, num_vo = get_shape(edges)
    edges = swap_rows(i+num_i-1, j+num_i-1, edges)
    return swap_cols(i-1, j-1, edges)


def swap_intermediates(i: int, j: int, edges: Array) -> Array:
    """
    Symmetry operation of the computational graph that interchanges the 
    naming of two vertices while preserving the computational graph.
    
    takes graph naming as input
    NOT jittable
    """
    i, j = lax.cond(i < j, 
                    lambda m, n: (m, n),
                    lambda m, n: (n, m),
                    i, j)
    
    num_i, num_vo = get_shape(edges)
    _i = i+num_vo-1
    _j = j+num_vo-1
    s1 = edges.at[:, _i].get()
    s2 = edges.at[:, _j].get()
    sum1 = jnp.sum(s1[_i+1:])
    sum2 = jnp.sum(s2[_i+1:])
        
    edges = lax.cond(sum1 + sum2 == 0,
                    lambda x: _swap_intermediates(i, j, x),
                    lambda x: x,
                    edges)
    
    return edges
    
    
    
