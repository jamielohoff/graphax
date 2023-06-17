from typing import Tuple

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..core import GraphInfo, make_empty_edges


def make_random(key: chex.PRNGKey,
                info: GraphInfo,
                fraction: float = .35) -> Tuple[chex.Array, GraphInfo]: 
    in_key, var_key, key = jrand.split(key, 3)
    
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    
    i_conns = jrand.uniform(in_key, (num_i, num_v))
    i_conns = jnp.where(i_conns > fraction, 0, 1)
    
    v_conns = jrand.uniform(var_key, (num_v, num_v))
    v_conns = jnp.where(v_conns > fraction, 0, 1)
    v_conns = jnp.triu(v_conns, k=1)
    
    edges = jnp.zeros((num_i+num_v, num_v))    
    edges = edges.at[:num_i, :].set(i_conns)
    edges = edges.at[num_i:, :num_v].set(v_conns)
    
    # Randomly select n <= info.num_outputs many output variables by deleting
    # the respective rows
    output_vertices = jnp.zeros(num_v)
    output_vertices, i = lax.cond(edges.at[:, -1].get().sum() != 0.,
                                lambda ov: (ov.at[0].set(num_v), 1),
                                lambda ov: (ov, 0),
                                output_vertices)
        
    n = jrand.randint(key, (), 1, num_o-i)
    idx = jrand.randint(key, (), 0, num_v)
    vertices = jrand.choice(key, jnp.arange(num_i+idx, num_i+num_v-i+1), (n,))
    
    def make_output_vertex_fn(carry, vertex):
        j, _edges, _output_vertices = carry
        _edges = _edges.at[vertex-1, :].set(0.)
        _output_vertices = _output_vertices.at[j].set(vertex-num_i)
        j += 1
        carry = (j, _edges, _output_vertices)
        return carry, None
    
    output, _ = lax.scan(make_output_vertex_fn, (i, edges, output_vertices), vertices)
    _, edges, output_vertices = output
    return edges, info, output_vertices


def make_connected_random(key: chex.PRNGKey,
                        info: GraphInfo,
                        p: chex.Array = None) -> Tuple[chex.Array, GraphInfo]:
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    edges = make_empty_edges(info)
    output_vertices = jnp.zeros(num_v)
    
    size_choices = jnp.arange(1, p.size+1)
    sizes = jrand.choice(key, size_choices, (num_v,), p=p)
    

    for j, size in zip(jnp.arange(0, num_v), sizes):
        subkey, key = jrand.split(key, 2)
        
        choices = jnp.arange(0, num_i+j)
        edge_positions = jrand.choice(subkey, choices, (size,), replace=False)
        def add_edge_fn(_edges, i):
            _edges = _edges.at[i, j].set(1.)
            return _edges, None
        edges, _ = lax.scan(add_edge_fn, edges, edge_positions)
        
    # TODO manage output graphs
        
    return edges, info, output_vertices
    
