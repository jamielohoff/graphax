import jax
import jax.numpy as jnp
import jax.tree_util as jtu


def tree_allclose(tree1, tree2):
    is_equal = jtu.tree_map(jnp.allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


def zeros_like(invar, outvar):
    in_shape = invar.aval.shape
    out_shape = outvar.aval.shape
    
    if in_shape == () and out_shape == ():
        return 0.
    else:
        shape = (*in_shape, *out_shape)
        return jnp.zeros_like(shape)
    

def add_slice(edges, outvar, idx, num_i, num_vo):
    """
    Function that adds another slice to the computational graph representation.
    This is useful for vertices that are output vertices but whose values are
    reused later in the computational graph as well.
    It effectively adds another vertex to the graph which is connected to
    the vertex in question by a single copy edge with sparsity type 8.
    """
    slc = jnp.zeros((5, num_i+num_vo+1), dtype=jnp.int32)
    
    if outvar.aval.shape == ():
        outvar_shape = (1, 1)
    elif len(outvar.aval.shape) == 1:
        outvar_shape = (outvar.aval.shape[0], 1)
    else:
        outvar_shape = outvar.aval.shape
    jac_shape = jnp.array([8, *outvar_shape, *outvar_shape])
    slc = slc.at[:, idx-num_i-1].set(jac_shape)
    
    edges = jnp.append(edges, slc[:, :, jnp.newaxis], axis=2)
    zeros = jnp.zeros((5, 1, num_vo+1), dtype=jnp.int32)
    edges = jnp.append(edges, zeros, axis=1)
    edges = edges.at[0, 0, 1].add(1)
    edges = edges.at[1, 0, -1].set(1)
    edges = edges.at[2, 0, -1].set(1)
    
    return edges
    

