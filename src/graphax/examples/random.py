import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from ..core import make_empty_edges


def make_random(key: PRNGKey, info: Array, p: Array = None):
    num_i, num_v, num_o = info
    edges = make_empty_edges(info)
    
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

    return edges
    
