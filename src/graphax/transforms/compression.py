import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array


def compress(edges: Array) -> Array:
    """
    Function that removes all zero rows and cols from a comp. graph repr.
    NOTE: This changes the shape of the edges array and the number of intermediate variables!
    """
    num_i, num_v, num_o = edges.at[0, 0, 0:3].get()
            
    i, num_removed_vertices = 1, 0
    for _ in range(1, num_v+1):            
        s1 = jnp.sum(edges.at[:, i+num_i, :].get()) == 0
        s2 = jnp.sum(edges.at[:, 1:, i-1].get()) == 0
        if s1 and s2:         
            edges = jnp.delete(edges, i+num_i, axis=1)
            edges = jnp.delete(edges, i-1, axis=2)
            num_removed_vertices += 1
        else:
            i += 1

    num_v = edges.shape[2]
    num_i = edges.shape[1] - num_v - 1
    num_o = jnp.sum(edges.at[2, 0, :].get())
    shape = jnp.array([num_i, num_v-num_o, num_o])
    edges = edges.at[0, 0, 0:3].set(shape)
    return edges

