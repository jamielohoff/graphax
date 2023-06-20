import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import chex

from ..core import GraphInfo, make_empty_edges


def make_random(key: chex.PRNGKey, info: GraphInfo, fraction: float = .35): 
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
    
    attn_mask = jnp.ones((num_v, num_v))
    
    output_vertices = jrand.choice(key, jnp.arange(1, num_v), (num_o-1,), replace=False)
    output_vertices = jnp.append(output_vertices, jnp.array([num_v]))
    
    # Manage output variables
    for j in jnp.arange(1, num_v):
        lenkey, vkey, key = jrand.split(key, 3)
        if j in output_vertices:
            print(j)
            edges = edges.at[j+num_i-1, :].set(0.)
        else:
            if jnp.sum(edges.at[j+num_i-1, :].get()) == 0.:
                num_edges = jrand.randint(lenkey, (), 1, 4)
                vertices = jrand.choice(vkey, jnp.arange(j, num_v), (num_edges,))
                for vertex in vertices:
                    edges = edges.at[j+num_i-1, vertex].set(1.)
    vertex_mask = jnp.append(output_vertices, jnp.zeros(num_v-num_o))
    return edges, info, vertex_mask, attn_mask


def make_connected_random(key: chex.PRNGKey, info: GraphInfo, p: chex.Array = None):
    num_i = info.num_inputs
    num_v = info.num_intermediates
    num_o = info.num_outputs
    edges = make_empty_edges(info)
    attn_mask = jnp.ones((num_v, num_v))
    
    size_choices = jnp.arange(1, p.size+1)
    sizes = jrand.choice(key, size_choices, (num_v,), p=p)
    choices = jnp.arange(0, num_i)
    
    output_vertices = jrand.choice(key, jnp.arange(1, num_v), (num_o-1,), replace=False)
    output_vertices = jnp.append(output_vertices, jnp.array([num_v]))
    
    # Populate edge matrix with a fully connected graph
    for j, size in zip(jnp.arange(1, num_v+1), sizes):
        subkey, key = jrand.split(key, 2)

        if not j in output_vertices:
            choices = jnp.append(choices, jnp.array([j+num_i-1]))
        edge_positions = jrand.choice(subkey, choices, (size,), replace=False)
        def add_edge_fn(_edges, i):
            _edges = _edges.at[i, j-1].set(1.)
            return _edges, None
        edges, _ = lax.scan(add_edge_fn, edges, edge_positions)
        
    # Manage output variables
    for j in jnp.arange(1, num_v):
        lenkey, vkey, key = jrand.split(key, 3)
        if j in output_vertices:
            edges = edges.at[j+num_i-1, :].set(0.)
        else:
            if jnp.sum(edges.at[j+num_i-1, :].get()) == 0.:
                num_edges = jrand.randint(lenkey, (), 1, p.size)
                vertices = jrand.choice(vkey, jnp.arange(j, num_v), (num_edges,))
                for vertex in vertices:
                    edges = edges.at[j+num_i-1, vertex].set(1.)

    vertex_mask = jnp.append(output_vertices, jnp.zeros(num_v-num_o))
    return edges, info, vertex_mask, attn_mask
    
