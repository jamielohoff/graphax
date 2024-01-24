from functools import partial

import numpy as np

import jax
import jax.lax as lax
import jax.numpy as jnp

import jax._src.core as core

from .sparse.tensor import (SparseTensor, 
                            DenseDimension, 
                            SparseDimension,
                            _swap_back_axes)


def get_ndim(arr):
    if type(arr) is float:
        return 0
    else:
        return arr.ndim
    
    
def get_shape(arr):
    if type(arr) is float:
        return ()
    else:
        return arr.shape
    

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
                primal_size = max(len(primal_dims), 1)
                if ps != os:
                    val_dim = sum([1 for d in out_dims if d.val_dim is not None])
                    out_dims.append(DenseDimension(i, os, val_dim))
                    primal_dims.append(DenseDimension(i+out_size+primal_size, ps, None))
                else:
                    val_dim = sum([1 for d in out_dims if d.size is not None])
                    out_dims.append(SparseDimension(i, os, val_dim, i+out_size+primal_size))
                    primal_dims.append(SparseDimension(i+out_size+primal_size, os, val_dim, i))
                for d in primal_dims[:-1]:
                    d.id += 1
                    if type(d) is SparseDimension:
                        _d = out_dims[d.other_id]
                        _d.other_id += 1

            return _swap_back_axes(SparseTensor(out_dims, primal_dims, elemental))
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
    print(primitive, elementals_out)
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

# TODO this can be made more efficient
def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))
    
defelemental(lax.add_p, add_elemental_rule)
defelemental(jax._src.ad_util.add_any_p, add_elemental_rule)

# TODO this can also be made more efficient
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


def eq_elemental_rule(x, y):
    return (jnp.zeros_like(y), jnp.zeros_like(x))
defelemental(lax.eq_p, eq_elemental_rule)


def pow_elemental_rule(out, x, y):
    return (y*x**(y-1), jnp.log(x)*out)
defelemental2(lax.pow_p, pow_elemental_rule)


def transpose_elemental_rule(primals, **params):
    val_out = lax.transpose_p.bind(*primals, **params)
    
    permutation = params["permutation"]
    elemental = jnp.ones_like(val_out)
    
    new_out_dims, new_primal_dims = [], []
    
    l = len(permutation)
    for i, p in enumerate(permutation):
        new_out_dims.insert(i, SparseDimension(i, elemental.aval.shape[i], i, l+p))
        new_primal_dims.insert(p, SparseDimension(l+p, elemental.aval.shape[i], i, i))
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, elemental)]

elemental_rules[lax.transpose_p] = transpose_elemental_rule


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
    new_out_dims, new_primal_dims = [], []
    _shape = []
                
    l = val_out.aval.ndim
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(idx, size, count))
            _shape.append(size)
            count += 1
        else:
            new_out_dims.append(SparseDimension(i, size, None, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, None, i))
            
    val = jnp.ones(_shape)
    print(SparseTensor(new_out_dims, new_primal_dims, val))
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]
    
elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)

    primal = primals[0]
    axes = params["axes"]
    
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif type(axes) is int:
        axes = (axes,)
    new_out_dims, new_primal_dims = [], []
    _shape = []
    
    l = val_out.aval.ndim
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            idx = len(new_out_dims) + len(new_primal_dims)
            new_primal_dims.append(DenseDimension(max(idx, 1), size, count))
            _shape.append(size)
            count += 1
        else:
            new_out_dims.append(SparseDimension(i, size, None, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, None, i))
            
    new_val = jnp.zeros(_shape)
    # TODO set to 1 at position of the maximum
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, new_val)]
    
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
    new_out_dims, new_primal_dims = [], []
    _shape = []
    
    l = val_out.aval.ndim
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            new_primal_dims.append(DenseDimension(max(i,1), size, count))
            _shape.append(size)
            count += 1
        else:
            new_out_dims.append(SparseDimension(i, size, None, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, None, i))
            
    val = jnp.zeros(_shape)
    # TODO set to 1 at position of the minimum
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]
    
elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule


def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    lhs, rhs = primals
    
    # Which dimensions of the tensors are contracted
    dimension_numbers = params["dimension_numbers"][0]
    # TODO correct batching if applicable
    
    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]
    
    # TODO properly treat the transpose
    if rhs.aval.ndim > 1:
        transpose_rhs = rhs.T
        # print(lhs, transpose_rhs)
        # TODO this needs generalization to higher-dimensional tensors
        transpose_rhs_dims = [1-d for d in dimension_numbers[1]]
    else:
        transpose_rhs = rhs
        transpose_rhs_dims = rhs_contracting_dims

    lhs_shape = list(lhs.aval.shape)
    rhs_shape = list(rhs.aval.shape)
    out_shape = list(val_out.aval.shape)
    
    lhs_jac_shape = out_shape + lhs_shape
    rhs_jac_shape = out_shape + rhs_shape
    
    lhs_primal_dims, rhs_primal_dims = [], []
    lhs_out_dims, rhs_out_dims = [], []
    
    i = 0
    for l, ld in enumerate(lhs_shape):
        _l = l + val_out.aval.ndim
        if l in lhs_contracting_dims:
            # Contracting dimension
            dim = transpose_rhs_dims[i] # TODO take account of the transpose
            lhs_primal_dims.append(DenseDimension(_l, transpose_rhs.aval.shape[dim], dim))
            i += 1
        else:
            lhs_primal_dims.append(SparseDimension(_l, lhs_jac_shape[i], None, l))
            lhs_out_dims.append(SparseDimension(l, lhs_jac_shape[i], None, _l))
            rhs_out_dims.append(DenseDimension(i, ld, l))
          
    j = 0   
    for r, rd in enumerate(rhs_shape):
        _r = r + len(val_out.aval.shape)
        if r in rhs_contracting_dims:
            # Contracting dimension
            dim = lhs_contracting_dims[j]
            rhs_primal_dims.append(DenseDimension(_r, lhs.aval.shape[dim], dim))
            j += 1
        else:
            rhs_primal_dims.append(SparseDimension(_r, rhs_jac_shape[j], None, r))
            rhs_out_dims.append(SparseDimension(r, rhs_jac_shape[j], None, _r))
            lhs_out_dims.append(DenseDimension(j, rd, transpose_rhs_dims.index(r)))
        
    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, transpose_rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)   
        
    return val_out, [lhs_tensor, rhs_tensor]

elemental_rules[lax.dot_general_p] = dot_general_elemental_rule


# TODO simplify this by terminating the edges instead of using zeros_like
def stop_gradient_elemental_rule(primals, **params):
    # Broadcasting adds sparse dimensions of size 1
    # and another sparse dimension for the variables
    val_out = lax.stop_gradient_p.bind(*primals, **params)
    
    primal = primals[0]    
    new_out_dims, new_primal_dims = [], []

    l = primal.ndim
    for i, size in enumerate(primal.shape):
        new_out_dims.append(SparseDimension(i, size, i, i+l))
        new_primal_dims.append(SparseDimension(i+l, size, i, i))

    val = jnp.zeros(primal.aval.shape)
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]
    
elemental_rules[lax.stop_gradient_p] = stop_gradient_elemental_rule


def iota_elemental_rule(primals, **params):
    val_out = lax.iota_p.bind(*primals, **params)
    return val_out, []

elemental_rules[lax.iota_p] = iota_elemental_rule


### Transforms

def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    
    def reshape_copy_gradient_fn(pre, post, iota):
        if post.val is None:
            pre.copy_gradient_fn = reshape_copy_gradient_fn
            return pre
        else:
            full_val = post.full(iota)
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
            post = SparseTensor(new_out_dims, new_primal_dims, full_val)
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [reshape_copy_gradient_fn])]

elemental_rules[lax.reshape_p] = reshape_elemental_rule


def slice_elemental_rule(primals, **params):
    val_out = lax.slice_p.bind(*primals, **params)
    start_indices = list(params["start_indices"])
    
    def slice_copy_gradient_fn(pre, post, iota):
        if post.val is None:
            pre.copy_gradient_fn = slice_copy_gradient_fn
            return pre
        else:
            full_val = post.full(iota)
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
            scatter_indices = jnp.array(start_indices, dtype=jnp.int32)
            scatter_indices = jnp.concatenate([scatter_zeros, scatter_indices])
            
            zeros = lax.scatter(zeros, scatter_indices, full_val, scatter_dims)
            post = SparseTensor(new_out_dims, new_primal_dims, zeros)
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [slice_copy_gradient_fn])]

elemental_rules[lax.slice_p] = slice_elemental_rule


def broadcast_elemental_rule(primals, **params):
    # TODO fix the case where we only have a single broadcast operation
    # Broadcasting adds DenseDimensions of size 1
    val_out = lax.broadcast_in_dim_p.bind(*primals, **params)
            
    # TODO This guy needs major revision
    # TODO rename pre and post into lhs, rhs 
    def broadcast_transform(pre, post, iota):
        if post.val is None:
            pre.jac_transform = [broadcast_transform]
            return pre
        else:
            print("old post", post)
            rm_dims = [d for d in range(val_out.ndim) 
                       if d not in params["broadcast_dimensions"]]

            new_out_dims = list(post.out_dims)
            new_primal_dims = list(post.primal_dims)
            _rm_dims = []
            for dim in rm_dims:
                if new_primal_dims[dim].val_dim is not None:
                    _rm_dims.append(new_primal_dims[dim].val_dim)
                if type(new_primal_dims[dim]) is DenseDimension:
                    del new_primal_dims[dim]
                
                    for d in new_primal_dims[dim:]:
                        d.id -= 1
                        if type(d) is SparseDimension:
                            _d = new_out_dims[d.other_id]
                            _d.other_id -= 1
                else:
                    id = new_primal_dims[dim].id
                    other_id = new_primal_dims[dim].other_id
                    del new_primal_dims[dim]
                    old_dim = new_out_dims[other_id]
                    new_out_dims[other_id] = DenseDimension(old_dim.id, old_dim.size, None)
                    for d in new_out_dims + new_primal_dims:
                        if d.id > id:
                            d.id -= 1
                            if type(d) is SparseDimension:
                                _d = new_out_dims[d.other_id]
                                _d.other_id -= 1
                    
            new_out_dims = tuple(new_out_dims)
            new_primal_dims = tuple(new_primal_dims)
            if len(_rm_dims) > 0:
                new_val = jnp.squeeze(post.val, axis=_rm_dims)
            else:
                new_val = post.val
            post = SparseTensor(new_out_dims, new_primal_dims, new_val, [])
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [broadcast_transform])]

    
elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule

