from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..graph import GraphState, add_edge

# TODO currently not jittable
def construct_layered_graph(ninputs: int, 
                            nintermediates: int, 
                            noutputs: int,
                            key: chex.PRNGKey, *,
                            low: int = 1, 
                            high: int = 4, 
                            intermediate_split: int = 2) -> Tuple[GraphState, int]:
    nedges = 0
    n = ninputs + nintermediates + noutputs
    edges = jnp.zeros((n, n), dtype=jnp.float32)
    state = jnp.zeros((nintermediates,))
    gs = GraphState(edges, 
                    state,
                    ninputs,
                    nintermediates,
                    noutputs)
    
    layer = ninputs
    next_layer = ninputs + nintermediates//intermediate_split
    
    for vertex in range(n-noutputs):
        int_key, val_key, key = jrand.split(key, 3)
                
        if vertex > layer:
            layer += nintermediates//intermediate_split
            next_layer += nintermediates//intermediate_split
            
        for _ in range(1, jrand.randint(int_key, (1,), minval=low, maxval=high)):
            other_vertex = jrand.randint(int_key, (1,), 
                                        minval=layer+1, 
                                        maxval=min(next_layer, n))
            gs = add_edge(gs, (vertex, other_vertex), 1.)
            nedges += 1
    return gs, nedges

# TODO currently not jittable
def construct_random_graph(ninputs: int, 
                            nintermediates: int, 
                            noutputs: int,
                            key: chex.PRNGKey, *, 
                            low: int = 1, 
                            high: int = 4) -> Tuple[GraphState, int]:
    
    nedges = 0
    n = ninputs + nintermediates + noutputs
    edges = jnp.zeros((n, n), dtype=jnp.float32)
    state = jnp.zeros((nintermediates,))
    gs = GraphState(edges, state)
    
    for vertex in range(n-noutputs):
        int_key, val_key, key = jrand.split(key, 3)
        num_edges = jrand.randint(int_key, (1,), minval=low, maxval=high)[0]
        other_vertices = jrand.randint(int_key, (num_edges,), 
                                    minval=vertex+1, 
                                    maxval=n)  
        for other_vertex in other_vertices:
            gs = add_edge(gs, (vertex, other_vertex), 1.)
            nedges += 1
    return gs, nedges

