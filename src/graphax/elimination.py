""" 
Edge elimination functions that are totally jit-compilable.
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
    """TODO add docstring

    Args:
        edge (Tuple[int]): _description_
    """
    num_inputs, _, _, num_edges, _ = info
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs.edges[e0, e1]
    gs.edges = gs.edges.at[e0, e1].set(0.)

    def front_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = gs.edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(i == edge[1], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, e0, nonzeros[1], val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, edges
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=num_edges,
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(front_update_edge, (gs.edges, 0), nonzeros)
    gs.edges = output[0]
    return gs, output[1]
 

def back_eliminate(gs: GraphState, 
                   edge: Tuple[int, int],
                   info: Array) -> Tuple[GraphState, int]:
    """TODO add docstring

    Args:
        edge (Tuple[int]): _description_
    """
    num_inputs, _, _, num_edges, _ = info
    
    e0 = edge[0] + num_inputs - 1
    e1 = edge[1] - 1
    
    edge_val = gs.edges[e0, e1]
    gs.edges = gs.edges.at[e0, e1].set(0.)

    def back_update_edge(carry, nonzeros):
        edges, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
        val = gs.edges.at[nonzeros[0], nonzeros[1]].get()
        edges, ops = lax.cond(j == edge[0], 
                            lambda x, m, n, val: (x.at[m, n].set(val), 1), 
                            lambda x, m, n, val: (x, 0), 
                            edges, nonzeros[0], e1, val*edge_val)
        nops += ops
        carry = (edges, nops)
        return carry, edges
    
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(back_update_edge, (gs.edges, 0), nonzeros)
    gs.edges = output[0]
    return gs, output[1]


def eliminate(gs: GraphState, 
            vertex: int, 
            info: Array) -> Tuple[GraphState, int]:
    """TODO add docstring

    Args:
        vertex (int): _description_
    """
    num_inputs, _, _, num_edges, _ = info
    
    def update_edges(carry, nonzeros):
        gs, nops = carry
        i = nonzeros[0] - num_inputs + 1
        j = nonzeros[1] + 1
                        
        gs, fops = lax.cond(j == vertex,
                            lambda x, m, n: front_eliminate(x, (m,n), info), 
                            lambda x, m, n: (x, 0), 
                            gs, i, j)

        gs, bops = lax.cond(i == vertex, 
                            lambda x, m, n: back_eliminate(x, (m,n), info), 
                            lambda x, m, n: (x, 0), 
                            gs, i, j)
        
        
        nops += (fops + bops)
        carry = (gs, nops)
        return carry, gs
        
    nonzeros = jnp.stack(jnp.nonzero(gs.edges, 
                                    size=num_edges, 
                                    fill_value=-num_edges)).transpose(1, 0)
    output, _ = lax.scan(update_edges, (gs, 0), nonzeros)
    gs = output[0]
    num_steps = gs.info[4]
    gs.info = gs.info.at[4].add(1)
    gs.state = gs.state.at[num_steps].set(vertex)
    return gs, output[1]


def forward(gs: GraphState, info: Array) -> Tuple[GraphState, int]:
    """TODO docstring

    Returns:
        _type_: _description_
    """
    _, num_intermediates, _, _, _ = info
    
    def fwd(carry, edge):
        gs, nops = carry
        gs, ops = eliminate(gs, edge, info)
        nops += ops
        carry = (gs, nops)
        return carry, edge
    vertex_list = jnp.arange(1, num_intermediates+1)
    output, _ = lax.scan(fwd, (gs, 0), vertex_list)
    gs = output[0]
    return gs, output[1]


def reverse(gs: GraphState, info: Array) -> Tuple[GraphState, int]:
    """TODO docstring

    Returns:
        _type_: _description_
    """
    _, num_intermediates, _, _, _ = info
    
    def rev(carry, edge):
        gs, nops = carry
        gs, ops = eliminate(gs, edge, info)
        nops += ops
        carry = (gs, nops)
        return carry, edge
    vertex_list = jnp.arange(1, num_intermediates+1)[::-1]
    output, _ = lax.scan(rev, (gs, 0), vertex_list)
    gs = output[0]
    return gs, output[1]

