from functools import reduce
import copy

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu


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
        out_size = reduce((lambda x, y: x*y), out_shape)
        val = jnp.eye(out_size).reshape(*out_shape, *primal_shape)
        
        # l = len(out_shape)
        # val = 1.
        # for i, o in enumerate(out_shape):
        #     j = primal_shape.index(o) # this does not work for repeated elements
        #     _shape = [1]*len(shape)
        #     _shape[i] = o
        #     _shape[l+j] = o
        #     if o == 1:
        #         kronecker = jnp.ones((1, 1)).reshape(_shape)
        #     else: 
        #         kronecker = jnp.eye(o).reshape(_shape)
        #     val *= kronecker
        return val


def eye_like_copy(shape, out_len, iota):
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    print("shape", shape)
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape)
        out_size = reduce((lambda x, y: x*y), out_shape)
        if out_size == 1:
            return jnp.ones((1,)+tuple(primal_shape))
        elif primal_size == 1:
            return jnp.ones(tuple(out_shape)+(1,))
        else:
            sub_iota = lax.slice(iota, (0, 0), (out_size, primal_size))
            return sub_iota.reshape(*shape)
    else:
        # This piece of code creates a proper higher order tensor as that is a
        # product of kronecker deltas
        # It does so by creating 2d tensors of the appropriate shape and then
        # reshaping them to the correct higher order shape and then multiplying
        # them together
        l = len(out_shape)
        val = 1.
        _primal_shape = copy.copy(primal_shape)
        for i, o in enumerate(out_shape):
            if o in primal_shape:
                j = primal_shape.index(o)
                _j = _primal_shape.index(o)
                primal_shape.pop(j)
                _shape = [1]*len(shape)
                _shape[i] = o
                _shape[l+_j] = o
                if o == 1:
                    kronecker = jnp.ones((1, 1)).reshape(_shape)
                else: 
                    sub_iota = lax.slice(iota, (0, 0), (o, o))
                    print(_shape)
                    kronecker = sub_iota.reshape(_shape)
                val *= kronecker
        return val
    
    
def get_largest_tensor(tensors):
    sizes = [t.aval.size for t in tensors]
    return max(sizes)    

