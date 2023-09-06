from functools import reduce

import jax
import jax.lax as lax
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
        return jnp.zeros(shape)
    

def eye_like(shape, out_len):
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape)
        out_size = reduce((lambda x, y: x*y), out_shape)
        if out_size == 1:
            return jnp.ones((1,)+tuple(primal_shape))
        elif primal_size == 1:
            return jnp.ones(tuple(out_shape)+(1,))
        else:
            return jnp.eye(out_size, primal_size).reshape(*out_shape, *primal_shape)
    else:
        l = len(out_shape)
        val = 1.
        for i, o in enumerate(out_shape):
            j = primal_shape.index(o)
            _shape = [1]*len(shape)
            _shape[i] = o
            _shape[l+j] = o
            if o == 1:
                kronecker = jnp.ones((1, 1)).reshape(_shape)
            else: 
                kronecker = jnp.eye(o).reshape(_shape)
            val *= kronecker
        return val


def eye_like_copy(shape, out_len, iota):
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape)
        out_size = reduce((lambda x, y: x*y), out_shape)
        if out_size == 1:
            return jnp.ones((1,)+tuple(primal_shape))
        elif primal_size == 1:
            return jnp.ones(tuple(out_shape)+(1,))
        else:
            sub_iota = lax.slice(iota, (0, 0), (out_size, primal_size))
            return sub_iota.reshape(*out_shape, *primal_shape)
    else:
        l = len(out_shape)
        val = 1.
        for i, o in enumerate(out_shape):
            j = primal_shape.index(o)
            _shape = [1]*len(shape)
            _shape[i] = o
            _shape[l+j] = o
            if o == 1:
                kronecker = jnp.ones((1, 1)).reshape(_shape)
            else: 
                sub_iota = lax.slice(iota, (0, 0), (o, o))
                kronecker = sub_iota.reshape(_shape)
            val *= kronecker
        return val
    
    
def get_largest_tensor(args):
    sizes = [arg.aval.size for arg in args]
    return max(sizes)


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
    

