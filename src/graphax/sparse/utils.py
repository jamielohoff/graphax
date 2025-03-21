import copy
from typing import Sequence
from functools import reduce

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax.core import JaxprEqn, ClosedJaxpr, ShapedArray


def zeros_like(invar: ShapedArray, outvar: ShapedArray) -> jnp.ndarray:
    """
    Function that creates an array of zeros. The shape of the array is the
    concatenation of the shapes of the input and output dimensions.

    Args:
        invar (ShapedArray): The input variable.
        outvar (ShapedArray): The output variable.

    Returns:
        jnp.ndarray: An array of zeros with the shape of the concatenation of the
                    shapes of the input and output dimensions.
    """
    in_shape = invar.aval.shape
    out_shape = outvar.aval.shape
    
    if in_shape == () and out_shape == ():
        return 0.
    else:
        shape = (*in_shape, *out_shape)
        return jnp.zeros(shape)
    

def eye_like(shape: Sequence[int], out_len: int) -> jnp.ndarray:
    """
    Function that creates a higher order tensor that is a product of Kronecker deltas.

    Args:
        shape (Sequence[int]): The shape of the higher order tensor that we want to create.
        out_len (int): The length of the output tensor, i.e. the number of 
                        output dimensions.
    
    Returns:
        jnp.ndarray: The higher order tensor that is a product of Kronecker deltas.
    """
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape, 1)
        out_size = reduce((lambda x, y: x*y), out_shape, 1)
        if out_size == 1:
            return jnp.ones((1,)+tuple(primal_shape))
        elif primal_size == 1:
            return jnp.ones(tuple(out_shape)+(1,))
        else:
            return jnp.eye(out_size, primal_size).reshape(*out_shape, *primal_shape)
    else:
        out_size = reduce((lambda x, y: x*y), out_shape, 1)
        val = jnp.eye(out_size).reshape(*out_shape, *primal_shape)
        return val


def eye_like_copy(shape: Sequence[int], out_len: int, iota: jnp.ndarray) -> jnp.ndarray:
    """
    Function that creates a higher order tensor that is a product of Kronecker deltas.
    It tries to reuse the identity matrix `iota` as much as possible to create the
    higher order tensor. If `iota` is too small, it creates a new identity matrix
    of the appropriate size.

    Args:
        shape (Sequence[int]): The shape of the higher order tensor that we want to create.
        out_len (int): The length of the output tensor, i.e. the number of 
                        output dimensions.
        iota (jnp.ndarray): The identity matrix that we use to create the higher 
                            order tensor.

    Returns:
        jnp.ndarray: The higher order tensor that is a product of Kronecker deltas.
    """
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape, 1)
        out_size = reduce((lambda x, y: x*y), out_shape, 1)
        if out_size == 1:
            return jnp.ones((1,)+tuple(primal_shape))
        elif primal_size == 1:
            return jnp.ones(tuple(out_shape)+(1,))
        else:
            if iota.shape[0] < out_size or iota.shape[1] < primal_size:
                iota = jnp.eye(max(out_size, primal_size))
            else:
                iota = lax.slice(iota, (0, 0), (out_size, primal_size))
            sub_iota = lax.slice(iota, (0, 0), (out_size, primal_size))
            return sub_iota.reshape(*shape)
    else:
        # This piece of code creates a proper higher order tensor as that is a
        # product of Kronecker deltas
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
                    if iota.shape[0] < o or iota.shape[1] < o:
                        sub_iota = jnp.eye(o)
                        kronecker = sub_iota.reshape(_shape)
                    else:
                        sub_iota = lax.slice(iota, (0, 0), (o, o))
                        kronecker = sub_iota.reshape(_shape)
                val *= kronecker # NOTE: This thing is crazy expensive to compute and not always necessary?
        return val
    
    
def get_largest_tensor(tensors: Sequence[ShapedArray]) -> int:
    """
    Function that computes the size of the largest tensor in a list of tensors.

    Args:
        tensors (Sequence): A list of tensors for which we want to know the size 
                            of the largest tensor.

    Returns:
        int: The size of the largest tensor in the list of tensors.
    """
    sizes = [t.aval.size for t in tensors]
    return max(sizes)   


def count_muls(eqn: JaxprEqn) -> int:
    """
    Function that counts the number of multiplications done by a jaxpr equation.
    The implementation treats every primitive as zero multiplications except for
    the `lax.dot_general` and `lax.mul` primitives. For these, simple algorithms
    for counting the number of multiplications are implemented.

    Args:
        eqn (core.JaxprEqn): The `JaxprEqn` of which we want to know how many
                            multiplications are happening.

    Returns:
        int: The number of multiplications done inside the jaxpr equation.
    """
    if eqn.primitive is lax.dot_general_p:
        contraction_dims = eqn.params["dimension_numbers"][0]
        batch_dims = eqn.params["dimension_numbers"][1]
        
        var0, var1 = eqn.invars
        var0_shape = list(var0.aval.shape)
        var1_shape = list(var1.aval.shape)
        
        for d in contraction_dims[1] + batch_dims[1]:
            var1_shape[d] = 1
        return reduce((lambda x, y: x*y), var0_shape, 1)*reduce((lambda x, y: x*y), var1_shape, 1)
    
    elif eqn.primitive is lax.mul_p:
        return reduce((lambda x, y: x*y), eqn.outvars[0].aval.shape, 1)
    else:
        return 0
    
    
def count_muls_jaxpr(jaxpr: ClosedJaxpr) -> int:
    """
    Function that counts the number of multiplications done within a jaxpr.

    Args:
        jaxpr (core.ClosedJaxpr): The `ClosedJaxpr` of which we want to know
                                how many multiplications are performed.
    
    Returns:
        int: The number of multiplications done within the jax
    """
    return sum([count_muls(eqn) for eqn in jaxpr.eqns])

