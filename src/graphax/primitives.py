from functools import partial
import copy

import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp

import jax._src.core as core

from .sparse.tensor import (SparseTensor, DenseDimension, SparseDimension, 
                            _swap_back_axes, get_num_muls)


def get_ndim(arr):
    if type(arr) is float or type(arr) is int:
        return 0
    else:
        return arr.ndim
    
    
def get_shape(arr):
    if type(arr) is float or type(arr) is int:
        return ()
    else:
        return arr.shape
    
    
def get_aval_shape(val):
    if type(val) is np.ndarray:
        return val.shape
    else:
        return val.aval.shape
    

# TODO simplify this
def make_parallel_jacobian(i, primals, val_out, elemental):
    primal = primals[i]
    primal_size = get_ndim(primal)
    out_size = get_ndim(val_out)

    if len(primals) == 1:
        if primal_size == 0 and out_size == 0:
            # Singletons
            out_dims = []
            primal_dims = []
        elif primal_size == 0:
            # Handling broadcast of singletons
            out_dims = [DenseDimension(i, e, i) for i, e in enumerate(val_out.aval.shape)]
            primal_dims = []
        else:
            out_dims = [SparseDimension(i, e, i, out_size+i) 
                            for i, e in enumerate(val_out.aval.shape)]
            primal_dims = [SparseDimension(out_size+i, e, i, i) 
                            for i, e in enumerate(val_out.aval.shape)]
    elif len(primals) == 2:
        if primal_size == 0 and out_size == 0:
            # Singletons
            out_dims = []
            primal_dims = []
        elif primal_size == 0:
            # Handling broadcast of singletons
            out_dims = [DenseDimension(i, e, i) for i, e in enumerate(val_out.aval.shape)]
            primal_dims = []
        elif get_shape(primals[i]) != get_shape(val_out):
            # Broadcasting case
            out_dims, primal_dims = [], []

            for i, (os, ps) in enumerate(zip(val_out.aval.shape, primal.aval.shape)):
                out_size = len(out_dims)
                primal_size = len(primal_dims)
                if ps != os:
                    val_dim = sum([1 for d in out_dims if d.val_dim is not None])
                    out_dims.append(DenseDimension(i, os, val_dim))
                    primal_dims.append(DenseDimension(out_size+primal_size+1, ps, None))
                else:
                    val_dim = sum([1 for d in out_dims if d.size is not None])
                    out_dims.append(SparseDimension(i, os, val_dim, out_size+primal_size+1))
                    primal_dims.append(SparseDimension(out_size+primal_size+1, os, val_dim, i))
                for d in primal_dims[:-1]:
                    d.id += 1
                    if type(d) is SparseDimension:
                        _d = out_dims[d.other_id]
                        _d.other_id += 1
            return _swap_back_axes(SparseTensor(out_dims, primal_dims, elemental))
        
        elif type(elemental) is float or elemental.size == 1:
            if type(elemental) is not float:
                elemental = jnp.squeeze(elemental) # TODO dirty quick fix that needs to be properly addressed
            out_dims = [SparseDimension(i, e, None, out_size+i)
                        for i, e in enumerate(primal.aval.shape)]
            primal_dims = [SparseDimension(out_size+i, e, None, i)
                            for i, e in enumerate(primal.aval.shape)]
        else:
            out_dims = [SparseDimension(i, e, i, out_size+i)
                        for i, e in enumerate(primal.aval.shape)]
            primal_dims = [SparseDimension(out_size+i, e, i, i)
                            for i, e in enumerate(primal.aval.shape)]
    else:
        raise NotImplementedError(f"Parallel Jacobians with {len(primals)} inputs not yet supported!")
        
    return SparseTensor(out_dims, primal_dims, elemental)


elemental_rules = {}


def defelemental(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental, elementalrule, primitive)


def standard_elemental(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(*primals, **params)
    elementals = elementals if type(elementals) is tuple else (elementals,)

    elementals_out = [make_parallel_jacobian(i, primals, val_out, elemental) 
                        for i, elemental in enumerate(elementals) 
                        if not type(primals[i]) in (float, np.ndarray, np.float32)]
    return val_out, elementals_out


# Useful for stuff such as exp_p
def defelemental2(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental2, elementalrule, primitive)


def standard_elemental2(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(val_out, *primals, **params)
    elementals = elementals if type(elementals) is tuple else (elementals,)
    elementals_out = [make_parallel_jacobian(i, primals, val_out, elemental)
                        for i, elemental in enumerate(elementals) 
                        if not type(primals[i]) in (float, np.ndarray, np.float32)]
    return val_out, elementals_out
    
    
# Define elemental partials
defelemental(lax.neg_p, lambda x: -jnp.ones_like(x))
defelemental2(lax.abs_p, lambda out, primal: primal/out) # NOTE: not differentiable here!
defelemental(lax.integer_pow_p, lambda x, y: y*x**(y-1))

defelemental2(lax.exp_p, lambda out, primal: out)
defelemental(lax.log_p, lambda x: 1./x)
defelemental2(lax.sqrt_p, lambda out, primal: .5/out)
defelemental2(lax.logistic_p, lambda out, primal: out*(1.-out))
defelemental(lax.log1p_p, lambda x: 1./(1.+ x))

defelemental(lax.sin_p, lambda x: jnp.cos(x))
defelemental(lax.asin_p, lambda x: 1./jnp.sqrt(1.0 - x**2))
defelemental(lax.cos_p, lambda x: -jnp.sin(x))
defelemental(lax.acos_p, lambda x: -1./jnp.sqrt(1.0 - x**2))
defelemental2(lax.tan_p, lambda out, primal: 1.+out**2)
defelemental(lax.atan_p, lambda x: 1./(1. + x**2))

defelemental(lax.sinh_p, lambda x: jnp.cosh(x))
defelemental(lax.asinh_p, lambda x: jnp.sqrt(1. + x**2))
defelemental(lax.cosh_p, lambda x: jnp.sinh(x))
defelemental(lax.acosh_p, lambda x: 1./jnp.sqrt(x**2 - 1.))
defelemental2(lax.tanh_p, lambda out, primal: 1.-out**2)
defelemental(lax.atanh_p, lambda x: 1./(1. - x**2))

defelemental(lax.erf_p, lambda x: 2.*jnp.exp(-x**2)/jnp.sqrt(jnp.pi))

# TODO this can be significantly optimized
def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))
defelemental(lax.add_p, add_elemental_rule)
defelemental(jax._src.ad_util.add_any_p, add_elemental_rule)


# TODO this can also be optimized significantly
def sub_elemental_rule(x, y):
    return (jnp.ones_like(y), -jnp.ones_like(x))
defelemental(lax.sub_p, sub_elemental_rule)

    
def mul_elemental_rule(x, y):
    return (y, x)
defelemental(lax.mul_p, mul_elemental_rule)
    

def div_elemental_rule(x, y):
    return (1./y, -x/y**2)
defelemental(lax.div_p, div_elemental_rule)


def atan2_elemental_rule(x, y):
    abs2 = (x**2+y**2)
    return (y/abs2, -x/abs2)
defelemental(lax.atan2_p, atan2_elemental_rule)


# TODO This needs the correct gradients
def eq_elemental_rule(x, y):
    return (jnp.zeros_like(y), jnp.zeros_like(x))
defelemental(lax.eq_p, eq_elemental_rule)


def pow_elemental_rule(out, x, y):
    return (y*x**(y-1), jnp.log(x)*out)
defelemental2(lax.pow_p, pow_elemental_rule)


# TODO Create a general reduce rule with a custom derivative!
def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)
        
    primal = primals[0]
    axes = params["axes"]
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0))
    elif type(axes) is int:
        axes = (axes,)
    new_out_dims, new_primal_dims, shape = [], [], []
                
    l = val_out.aval.ndim
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            # idx = len(new_out_dims) + len(new_primal_dims)
            # idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(l+i, size, count))
            shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, None, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, None, ll))
            
    val = jnp.ones(shape, dtype=jnp.float32)
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]
    
elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)

    primal = primals[0]
    axes = params["axes"]
    shape = list(val_out.aval.shape)
    
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif type(axes) is int:
        axes = (axes,)
    new_out_dims, new_primal_dims, _shape = [], [], []
    
    l = val_out.aval.ndim
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            shape.insert(i, 1)
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(idx, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, i, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, i, ll))
            
    _val_out = val_out.reshape(shape)
    new_val = jnp.where(primal == _val_out, 1, 0) 
    # NOTE: Normalization is important if the maximum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm 

    return val_out, [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]
    
elemental_rules[lax.reduce_max_p] = reduce_max_elemental_rule


def reduce_min_elemental_rule(primals, **params):
    val_out = lax.reduce_min_p.bind(*primals, **params)
    
    primal = primals[0]
    axes = params["axes"]
    
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif type(axes) is int:
        axes = (axes,)
    new_out_dims, new_primal_dims, _shape = [], [], []
    
    l = val_out.aval.ndim
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(idx, size, i))
            _shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, i, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, i, ll))
            
    new_val = jnp.where(primal == val_out, 1, 0) 
    # NOTE: Normalization is important if the minimum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val/norm
    return val_out, [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]
    
elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule


def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    lhs, rhs = primals
    
    # Which dimensions of the tensors are contracted
    dimension_numbers = params["dimension_numbers"][0]
    batch_dims = params["dimension_numbers"][1]
    # NOTE: Batch dimensions are just treated as SparseDimensions.
    # However, currently the implementation only supports a single batch dimension
    
    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]
    
    lhs_batch_dims = batch_dims[0]
    rhs_batch_dims = batch_dims[1]
    
    lhs_shape = list(get_aval_shape(lhs))
    rhs_shape = list(get_aval_shape(rhs))
    out_shape = list(get_aval_shape(val_out))
        
    lhs_out_dims, rhs_out_dims = [], []
    lhs_primal_dims, rhs_primal_dims = [], []
        
    i, ii = 0, 0
    batch_dim_counter = 0
    for lid, ld in enumerate(lhs_shape):
        other_lid = lid + len(out_shape)
        if lid in lhs_contracting_dims:
            # Contracting dimension
            dim = rhs_contracting_dims[i]
            lhs_primal_dims.append(DenseDimension(other_lid, rhs_shape[dim], dim))
            i += 1
        else:
            if lid in lhs_batch_dims:
                # If it is a batch dimension, we need to treat it as a SparseDimension
                # with a valid `val_dim`
                dim = rhs_batch_dims[ii]
                ii += 1
                # TODO do not just use 0 here so that we can use multiple batch dimensions
                lhs_out_dims.insert(batch_dim_counter, SparseDimension(batch_dim_counter, ld, dim, other_lid))
                lhs_primal_dims.append(SparseDimension(other_lid, ld, dim, batch_dim_counter))
                batch_dim_counter += 1
                for d in lhs_out_dims[batch_dim_counter:]:
                    d.id += 1
            else:
                # Otherwise, we can just set `val_dim`` to None
                lhs_out_dims.append(SparseDimension(len(lhs_out_dims), ld, None, other_lid))
                lhs_primal_dims.append(SparseDimension(other_lid, ld, None, len(lhs_out_dims)-1))
                rhs_out_dims.append(DenseDimension(len(rhs_out_dims), ld, lid)) # TODO need to use insert here!
          
    j, jj = 0, 0   
    batch_dim_counter = 0
    for rid, rd in enumerate(rhs_shape):
        other_rid = rid + len(out_shape)
        if rid in rhs_contracting_dims:
            # Contracting dimension
            dim = lhs_contracting_dims[j]
            rhs_primal_dims.append(DenseDimension(other_rid, lhs_shape[dim], dim))
            j += 1
        else:
            if rid in rhs_batch_dims:
                # If it is a batch dimension, we need to treat it as a SparseDimension
                # with a valid `val_dim`
                dim = lhs_batch_dims[jj]
                jj += 1
                rhs_out_dims.insert(batch_dim_counter, SparseDimension(batch_dim_counter, rd, dim, other_rid))
                rhs_primal_dims.append(SparseDimension(other_rid, rd, dim, batch_dim_counter))
                batch_dim_counter += 1
                for d in rhs_out_dims[batch_dim_counter:]:
                    d.id += 1
            else:
                # Otherwise, we can just set `val_dim`` to None
                rhs_out_dims.append(SparseDimension(len(rhs_out_dims), rd, None, other_rid))
                rhs_primal_dims.append(SparseDimension(other_rid, rd, None, len(rhs_out_dims)-1))
                lhs_out_dims.append(DenseDimension(len(lhs_out_dims), rd, rid)) # TODO need to use insert here!
        
    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)   
        
    lhs_tensor = _swap_back_axes(lhs_tensor)
    rhs_tensor = _swap_back_axes(rhs_tensor)
    return val_out, [lhs_tensor, rhs_tensor]

elemental_rules[lax.dot_general_p] = dot_general_elemental_rule


def iota_elemental_rule(primals, **params):
    val_out = lax.iota_p.bind(*primals, **params)
    return val_out, []

elemental_rules[lax.iota_p] = iota_elemental_rule


def device_put_elemental_rule(primals, **params):
    val_out = lax.device_put_p.bind(*primals, **params)
    return val_out, []

elemental_rules[lax.device_put_p] = device_put_elemental_rule


### Transforms
from typing import Callable

Transform = Callable[[SparseTensor, SparseTensor, jnp.ndarray | None], SparseTensor]

class JacobianTransform:
    transform: Transform | None
    inverse_transform: Transform | None
    
    def __init__(self, transform: Transform, inverse_transform: Transform | None = None) -> None:
        self.transform = transform
        self.inverse_transform = inverse_transform
        
    def __repr__(self) -> str:
        return f"JacobianTransform(transform={self.transform}, " \
                f"inverse_transform={self.inverse_transform})"
        
    def apply(self, tensor: SparseTensor, iota: jnp.ndarray | None) -> SparseTensor:
        if self.inverse_transform is None:
            raise NotImplementedError("Transform not implemented!")
        return self.transform(tensor, iota)
    
    def apply_inverse(self, tensor: SparseTensor, iota: jnp.ndarray | None) -> SparseTensor:
        if self.inverse_transform is None:
            raise NotImplementedError("Inverse transform not implemented!")
        return self.inverse_transform(tensor, iota)


# TODO does not work as intended for high-dimensional stuff
def transpose_elemental_rule(primals, **params):
    # The slice primitive is written in such a way that it just densifies the
    # Jacobian and then slices it. This is not efficient and there might be ways
    # to make this more efficient by checking if sparse dimensions are untouched
    # how this changes the Jacobian.
    val_out = lax.transpose_p.bind(*primals, **params)
    permutation = params["permutation"]
    
    def transpose_transform(post, iota):
        new_out_dims = post.out_dims
        new_primal_dims = []
        counter = len(post.out_dims)
        
        for p in permutation:
            new_primal_dims.append(post.primal_dims[p])
            new_primal_dims[-1].id = counter
            if type(new_primal_dims[-1]) is SparseDimension:
                other_id = new_primal_dims[-1].other_id
                new_out_dims[other_id].other_id = counter
            counter += 1   
        return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, post.val))  
    return val_out, [SparseTensor([], [], None, [], [transpose_transform])]


# Should work for high-dimensional stuff
def alt_transpose_elemental_rule(primals, **params):
    # This primitive is written such that it applies the transpose to the out_dims 
    # of the pre_tensor
    val_out = lax.transpose_p.bind(*primals, **params)
    permutation = params["permutation"]
    
    def transpose_transform(pre, iota):
        new_out_dims = []
        new_primal_dims = pre.primal_dims
        counter = 0
        l = len(pre.out_dims)
                
        for p in permutation:
            new_out_dims.append(pre.out_dims[p])
            new_out_dims[-1].id = counter
            if type(new_out_dims[-1]) is SparseDimension:
                other_id = new_out_dims[-1].other_id
                new_primal_dims[other_id-l].other_id = counter
            counter += 1   

        return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, pre.val))  
    
    def inverse_transpose_transform(pre, post, iota):
        new_out_dims = post.out_dims
        new_primal_dims = []
        counter = len(post.out_dims)
        
        for p in permutation:
            new_primal_dims.append(post.primal_dims[p])
            new_primal_dims[-1].id = counter
            if type(new_primal_dims[-1]) is SparseDimension:
                other_id = new_primal_dims[-1].other_id
                new_out_dims[other_id].other_id = counter
            counter += 1   
        return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, post.val))
    transform = JacobianTransform(transpose_transform, inverse_transpose_transform)
    return val_out, [SparseTensor([], [], None, [transform], [])]

elemental_rules[lax.transpose_p] = alt_transpose_elemental_rule


def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    
    # TODO we have a lot of cases to catch here if we want to make this efficient!
    
    def reshape_transform(pre, post, iota):
        full_val = post.dense(iota)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseDimension(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseDimension(counter, s, counter))
            new_shape.append(s)
            counter += 1
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)
    return val_out, [SparseTensor([], [], None, [], [reshape_transform])]


def alt_reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    
    # TODO: dimensional collapse is not covered here!
    # Implement sparsity-aware version for significant speedup!
    
    def reshape_transform(pre, iota):
        full_val = pre.dense(iota) # NOTE array is not correctly materialized sometimes!
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        
        for s in val_out.shape:
            new_out_dims.append(DenseDimension(counter, s, counter))
            new_shape.append(s)
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseDimension(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
            
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)
    
    def inverse_reshape_transform(post, iota):
        full_val = post.dense(iota)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseDimension(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseDimension(counter, s, counter))
            new_shape.append(s)
            counter += 1
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)
    
    transform = JacobianTransform(reshape_transform, inverse_reshape_transform)
    return val_out, [SparseTensor([], [], None, [transform], [])]


elemental_rules[lax.reshape_p] = alt_reshape_elemental_rule


def slice_elemental_rule(primals, **params):
    # NOTE: rule is stupid like this because it increases computational cost
    # The slice primitive is written in such a way that it just densifies the
    # Jacobian and then scatters it. This is not efficient and there might be ways
    # to make this more efficient by checking if sparse dimensions are untouched
    # how this changes the Jacobian.
    val_out = lax.slice_p.bind(*primals, **params)
    start_indices = list(params["start_indices"])
    limit_indices = list(params["limit_indices"])
    
    def slice_transform(pre, post, iota):
        full_val = post.dense(iota)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseDimension(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        scatter_zeros = jnp.zeros(counter, dtype=jnp.int32)
        
        for s in primals[0].shape:
            new_primal_dims.append(DenseDimension(counter, s, counter))
            new_shape.append(s)
            counter += 1

        zeros = jnp.zeros(new_shape)
        dims = list(range(zeros.ndim))
        scatter_dims = lax.ScatterDimensionNumbers(dims, [], dims)
        _scatter_indices = jnp.array(start_indices, dtype=jnp.int32)
        scatter_indices = jnp.concatenate([scatter_zeros, _scatter_indices])

        new_val = lax.scatter(zeros, scatter_indices, full_val, scatter_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    return val_out, [SparseTensor([], [], None, [], [slice_transform])]


def alt_slice_elemental_rule(primals, **params):
    # The slice primitive is written in such a way that it just densifies the
    # Jacobian and then slices it. This is not efficient and there might be ways
    # to make this more efficient by checking if sparse dimensions are untouched
    # how this changes the Jacobian.
    val_out = lax.slice_p.bind(*primals, **params)
    
    def slice_transform(pre, iota):
        start_indices = list(params["start_indices"])
        limit_indices = list(params["limit_indices"])
        full_val = pre.dense(iota)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseDimension(counter, s, counter))
            counter += 1
        
        for d in pre.primal_dims:
            new_primal_dims.append(DenseDimension(counter, d.size, counter))
            start_indices.append(0)
            limit_indices.append(d.size)
            counter += 1

        new_val = lax.slice(full_val, start_indices, limit_indices)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    def inverse_slice_transform(post, iota):
        start_indices = list(params["start_indices"])
        limit_indices = list(params["limit_indices"])
        full_val = post.dense(iota)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0
        for d in post.out_dims:
            new_out_dims.append(DenseDimension(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        scatter_zeros = jnp.zeros(counter, dtype=jnp.int32)
        
        for s in primals[0].shape:
            new_primal_dims.append(DenseDimension(counter, s, counter))
            new_shape.append(s)
            counter += 1

        zeros = jnp.zeros(new_shape)
        dims = list(range(zeros.ndim))
        scatter_dims = lax.ScatterDimensionNumbers(dims, [], dims)
        _scatter_indices = jnp.array(start_indices, dtype=jnp.int32)
        scatter_indices = jnp.concatenate([scatter_zeros, _scatter_indices])

        new_val = lax.scatter(zeros, scatter_indices, full_val, scatter_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    transform = JacobianTransform(slice_transform, inverse_slice_transform)
    return val_out, [SparseTensor([], [], None, [transform], [])]


elemental_rules[lax.slice_p] = alt_slice_elemental_rule


def broadcast_elemental_rule(primals, **params):
    # TODO fix the case where we only have a single broadcast operation
    # Broadcasting adds DenseDimensions of size 1
    val_out = lax.broadcast_in_dim_p.bind(*primals, **params)
            
    # TODO This guy needs major revision
    # TODO rename pre and post into lhs, rhs 
    def inverse_broadcast_transform(post, iota):
        rm_dims = [d for d in range(val_out.ndim) 
                    if d not in params["broadcast_dimensions"]]

        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        _rm_dims = []
        for dim in rm_dims:
            if new_primal_dims[dim].val_dim is not None:
                _rm_dims.append(new_primal_dims[dim].val_dim)
            if type(new_primal_dims[dim]) is DenseDimension:
                has_smaller_dims = sum([1 for d in new_primal_dims[:dim+1] if d.val_dim is not None]) > 0
                del new_primal_dims[dim]
                for d in new_primal_dims[dim:]:
                    d.id -= 1
                    if type(d) is SparseDimension:
                        _d = new_out_dims[d.other_id]
                        _d.other_id -= 1
                    
            else:
                id = new_primal_dims[dim].id
                other_id = new_primal_dims[dim].other_id
                old_dim = new_out_dims[other_id]
                new_out_dims[other_id] = DenseDimension(old_dim.id, old_dim.size, None)
                has_smaller_dims = sum([1 for d in new_primal_dims[:dim+1] if d.val_dim is not None]) > 0
                del new_primal_dims[dim]
                for d in new_out_dims + new_primal_dims:
                    if d.id > id:
                        d.id -= 1
                        if type(d) is SparseDimension:
                            _d = new_out_dims[d.other_id]
                            _d.other_id -= 1
                            if d.val_dim is not None and has_smaller_dims:
                                d.val_dim -= 1
                                _d.val_dim -= 1
                        else:
                            if d.val_dim is not None and has_smaller_dims:
                                d.val_dim -= 1
                
        new_out_dims = tuple(new_out_dims)
        new_primal_dims = tuple(new_primal_dims)
        if len(_rm_dims) > 0:
            if all([post.val.shape[d] == 1 for d in _rm_dims]):
                new_val = jnp.squeeze(post.val, axis=_rm_dims)
            else:
                new_val = jnp.sum(post.val, axis=_rm_dims)
        else:
            new_val = post.val
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    transform = JacobianTransform(None, inverse_broadcast_transform)
    return val_out, [SparseTensor([], [], None, [], [transform])]

elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule


def squeeze_elemental_rule(primals, **params):
    # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
    # since it just adds a DenseDimension of size 1
    val_out = lax.squeeze_p.bind(*primals, **params)
    
    def squeeze_transform(pre, post, iota):
        new_dims = params["dimensions"]
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        for dim in new_dims:
            val_dim = sum([1 for d in new_out_dims if d.val_dim is not None])
            val_dim += sum([1 for d in new_primal_dims[:dim] if d.val_dim is not None and type(d) is DenseDimension])
            new_primal_dims.insert(dim, DenseDimension(dim, 1, val_dim))
            for d in new_primal_dims[dim:]:
                d.id += 1
                if d.val_dim is not None:
                    d.val_dim += 1
                if type(d) is SparseDimension:
                    _d = new_out_dims[d.other_id]
                    _d.other_id += 1
                    if _d.val_dim is not None:
                        _d.val_dim += 1
            
        new_val = jnp.expand_dims(post.val, axis=new_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    return val_out, [SparseTensor([], [], None, [], [squeeze_transform])]


def alt_squeeze_elemental_rule(primals, **params):
    # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
    # since it just adds a DenseDimension of size 1
    val_out = lax.squeeze_p.bind(*primals, **params)
    
    def squeeze_transform(pre, iota):
        dims = sorted(params["dimensions"])
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        squeeze_dims = []
        counter = 0
        
        for id in dims:
            idx = [j for j, d in enumerate(new_out_dims) if d.id == id][0]
            val_dim = new_out_dims[idx].val_dim
            squeeze_dims.append(val_dim)
            
            if type(new_out_dims[idx]) is SparseDimension:
                def _check(d, id):
                    if type(d) is SparseDimension:
                        return d.other_id == id
                    else:
                        return False
                other_idx = [j for j, d in enumerate(new_primal_dims) if _check(d, id)][0]
                other_dim = new_primal_dims[other_idx]
                new_primal_dims[other_idx] = DenseDimension(other_dim.id, other_dim.size, None)
                
            del new_out_dims[idx]
            counter += 1
            
        out_ids = [d.id for d in new_out_dims]
        primal_ids = [d.id for d in new_primal_dims]    
        new_val_dims = [d.val_dim for d in new_out_dims if d.val_dim is not None]
        new_val_dims += [d.val_dim for d in new_primal_dims if type(d) is DenseDimension and d.val_dim is not None]

        for d in new_out_dims:
            d.id = out_ids.index(d.id)
            if d.val_dim is not None:
                d.val_dim = new_val_dims.index(d.val_dim)
                if type(d) is SparseDimension:
                    d.other_id = len(new_out_dims) + new_primal_dims.index(d.other_id)
        for d in new_primal_dims:
            d.id = len(new_out_dims) + primal_ids.index(d.id)
            if d.val_dim is not None:
                d.val_dim = new_val_dims.index(d.val_dim)
                if type(d) is SparseDimension:
                    d.other_id = new_out_dims.index(d.other_id)
                    
        squeeze_dims = [d for d in squeeze_dims if d is not None]
        if len(squeeze_dims) > 0:
            new_val = jnp.squeeze(pre.val, axis=squeeze_dims)
        else:
            new_val = pre.val
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    def inverse_squeeze_transform(post, iota):
        new_dims = params["dimensions"]
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        for dim in new_dims:
            val_dim = sum([1 for d in new_out_dims if d.val_dim is not None])
            val_dim += sum([1 for d in new_primal_dims[:dim] if d.val_dim is not None and type(d) is DenseDimension])
            new_primal_dims.insert(dim, DenseDimension(dim, 1, val_dim))
            for d in new_primal_dims[dim:]:
                d.id += 1
                if d.val_dim is not None:
                    d.val_dim += 1
                if type(d) is SparseDimension:
                    _d = new_out_dims[d.other_id]
                    _d.other_id += 1
                    if _d.val_dim is not None:
                        _d.val_dim += 1
            
        new_val = jnp.expand_dims(post.val, axis=new_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    transform = JacobianTransform(squeeze_transform, inverse_squeeze_transform)
    return val_out, [SparseTensor([], [], None, [transform], [])]


elemental_rules[lax.squeeze_p] = alt_squeeze_elemental_rule


from .sparse.tensor import _materialize_dimensions


def concatenate_elemental_rule(primals, **params):
    # This gradient transformation is designed to take an post edge and
    # decompose it into the pre edges. This is done by densifying the post along
    # the respective axes and then use jnp.split to split the tensor.
    val_out = lax.concatenate_p.bind(*primals, **params)
    dim = params["dimension"]
    
    count = primals[0].shape[dim]
    slices = {primals[0]: [0, primals[0].shape[dim]]}
    _count = primals[0].shape[dim]
    for val in primals[1:]:
        count += val.shape[dim]
        slices[val] = [_count, count]
        _count = count
    
    def inverse_concatenate_transform(primal, post, iota):
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))

        d = new_primal_dims[dim]
        if type(d) is DenseDimension:
            if d.val_dim is not None:
                new_val = lax.slice_in_dim(post.val, *slices[primal], axis=d.val_dim)
                d.size = new_val.shape[d.val_dim]
            else:
                raise NotImplementedError("DenseDimension without `val_dim` not yet supported!")
        else:
            _d = new_out_dims[d.other_id]
            if d.val_dim is not None:
                new_out_dims[d.other_id] = DenseDimension(_d.id, _d.size, _d.val_dim)
                size = slices[primal][1] - slices[primal][0]
                
                # Calculate the new val_dim of the primal dimension
                val_dim = sum([1 for d in new_out_dims if d.val_dim is not None])
                val_dim += sum([1 for d in new_primal_dims[:dim] if d.val_dim is not None and type(d) is DenseDimension])
                new_primal_dims[dim] = DenseDimension(_d.other_id, size, val_dim)
                
                # Update the val_dim of all following dimensions
                for d in new_primal_dims[dim+1:]:
                    if type(d) is DenseDimension:
                        if d.val_dim is not None:
                            d.val_dim += 1
                
                # The following piece of code materialized the particular set
                # of sparse dimensions related to the concatenation dimension
                new_val = _materialize_dimensions(post, [d.id])
                
                if iota.shape[0] < d.size or iota.shape[1] < d.size:
                    sub_iota = jnp.eye(d.size, dtype=jnp.float32)
                else:
                    sub_iota = lax.slice(iota, [0, 0], [d.size, d.size])
                    
                shape = [1 for _ in range(post.val.ndim)]
                shape[_d.val_dim] = _d.size
                shape.insert(val_dim, d.size)
                sub_iota = sub_iota.reshape(shape)
                                
                new_val = new_val * sub_iota
                
                new_val = lax.slice_in_dim(new_val, *slices[primal], axis=val_dim)
                d.size = new_val.shape[d.val_dim]
                _d.size = new_val.shape[d.val_dim]
            else:
                raise NotImplementedError("Finish the implementation!")
                _d = new_out_dims[d.other_id]
                if d.val_dim is not None:
                    size = slices[primal][1] - slices[primal][0]
                    
                    out_val_dim = sum([1 for d in new_out_dims[:d.other_id] if d.val_dim is not None])
                    primal_val_dim = sum([1 for d in new_out_dims if d.val_dim is not None])
                    primal_val_dim += sum([1 for d in new_primal_dims[:dim] if d.val_dim is not None and type(d) is DenseDimension])
                    
                    new_out_dims[d.other_id] = DenseDimension(_d.id, _d.size, out_val_dim)
                    new_primal_dims[dim] = DenseDimension(_d.other_id, size, primal_val_dim)
                    
                    # TODO finish this!
                    for d in new_out_dims[d.other_id:]:
                        if type(d) is DenseDimension:
                            if d.val_dim is not None:
                                d.val_dim += 1
                    
                    # increase the val_dim of all following dimensions
                    for d in new_primal_dims[dim+1:]:
                        if type(d) is DenseDimension:
                            if d.val_dim is not None:
                                d.val_dim += 1
                    
                    # The following piece of code materialized the particular set
                    # of sparse dimensions related to the concatenation dimension
                    new_val = _materialize_dimensions(post, [d.id, d.other_id])
                    
                    if iota.shape[0] < d.size or iota.shape[1] < d.size:
                        sub_iota = jnp.eye(d.size, dtype=jnp.float32)
                    else:
                        sub_iota = lax.slice(iota, [0, 0], [d.size, d.size])
                        
                    shape = [1 for _ in range(post.val.ndim)]
                    shape.insert(out_val_dim, _d.size)
                    shape.insert(primal_val_dim, size)
                    sub_iota = sub_iota.reshape(shape)
                                        
                    new_val = new_val * sub_iota
                    
                    new_val = lax.slice_in_dim(new_val, *slices[primal], axis=primal_val_dim)
                    d.size = new_val.shape[d.val_dim]
                    _d.size = new_val.shape[d.val_dim]
        return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    return val_out, [SparseTensor([], [], None, [], [JacobianTransform(None, partial(inverse_concatenate_transform, p))]) for p in primals]

elemental_rules[lax.concatenate_p] = concatenate_elemental_rule

    
def convert_element_type_rule(primals, **params):
    val_out = lax.convert_element_type_p.bind(*primals, **params)
    new_dtype = params["new_dtype"]

    def convert_element_type_transform(pre, post, iota):
        new_post_val = lax.convert_element_type(post.val, new_dtype)
        return post.copy(val=new_post_val)
    return val_out, [SparseTensor([], [], None, [], [convert_element_type_transform])]


def alt_convert_element_type_rule(primals, **params):
    # TODO check if this is actually correct
    val_out = lax.convert_element_type_p.bind(*primals, **params)
    new_dtype = params["new_dtype"]

    def convert_element_type_transform(pre, iota):
        new_pre_val = lax.convert_element_type(pre.val, new_dtype)
        return pre.copy(val=new_pre_val)
    
    def inverse_convert_element_type_transform(post, iota):
        new_post_val = lax.convert_element_type(post.val, new_dtype)
        return post.copy(val=new_post_val)
    
    transform = JacobianTransform(convert_element_type_transform, inverse_convert_element_type_transform)
    return val_out, [SparseTensor([], [], None, [transform])]

    
elemental_rules[lax.convert_element_type_p] = alt_convert_element_type_rule

