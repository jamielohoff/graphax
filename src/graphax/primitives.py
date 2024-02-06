from functools import partial
import copy

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
# TODO implement this such that it becomes more efficient, i.e. through the
# use of jac_transform
# defelemental2(lax.stop_gradient_p, lambda out, primal: jnp.zeros_like(out))

def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))
    
defelemental(lax.add_p, add_elemental_rule)
defelemental(jax._src.ad_util.add_any_p, add_elemental_rule)

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
    
    lhs_shape = list(lhs.aval.shape)
    rhs_shape = list(rhs.aval.shape)
    out_shape = list(val_out.aval.shape)
        
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


### Transforms
def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    
    # TODO we have a lot of cases to catch here if we want to make this efficient!
    
    def reshape_transform(pre, post, iota):
        if post.val is None:
            pre.copy_gradient_fn = reshape_transform
            return pre
        else:
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
            post = SparseTensor(new_out_dims, new_primal_dims, full_val)
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [reshape_transform])]

elemental_rules[lax.reshape_p] = reshape_elemental_rule


def slice_elemental_rule(primals, **params):
    val_out = lax.slice_p.bind(*primals, **params)
    start_indices = list(params["start_indices"])
    limit_indices = list(params["limit_indices"])
    
    def slice_transform(pre, post, iota):
        if post.val is None:
            pre.jac_transform = slice_transform
            return pre
        else:
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
            scatter_indices = jnp.array(start_indices, dtype=jnp.int32)
            scatter_indices = jnp.concatenate([scatter_zeros, scatter_indices])
            
            zeros = lax.scatter(zeros, scatter_indices, full_val, scatter_dims)
            post = SparseTensor(new_out_dims, new_primal_dims, zeros)
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [slice_transform])]

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
                new_val = jnp.squeeze(post.val, axis=_rm_dims)
            else:
                new_val = post.val
                print("new post", post)
            post = SparseTensor(new_out_dims, new_primal_dims, new_val, [])
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [broadcast_transform])]

elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule


def squeeze_elemental_rule(primals, **params):
    # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
    # since it just adds a DenseDimension of size 1
    val_out = lax.squeeze_p.bind(*primals, **params)
    
    def squeeze_transform(pre, post, iota):
        if post.val is None:
            pre.jac_transform = [squeeze_transform]
            return pre
        else:
            new_dims = params["dimensions"]

            print("old post", post)
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
            post = SparseTensor(new_out_dims, new_primal_dims, new_val, [])
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [squeeze_transform])]

elemental_rules[lax.squeeze_p] = squeeze_elemental_rule


# def concatenate_elemental_rule(primals, **params):
#     # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
#     # since it just adds a DenseDimension of size 1
#     val_out = lax.concatenate_p.bind(*primals, **params)
    
#     print(params)
#     dim = params["dimension"]
    
#     count = primals[0].shape[dim]
#     slices = {primals[0]: [0, primals[0].shape[dim]]}
#     _count = primals[0].shape[dim]
#     for val in primals[1:]:
#         count += val.shape[dim]
#         slices[val] = [_count, count]
#         _count = count
#     print(slices)
    
#     def concatenate_transform(pre, post, iota):
#         if post.val is None:
#             pre.jac_transform = [concatenate_transform]
#             return pre
#         else:
#             new_out_dims = list(copy.deepcopy(post.out_dims))
#             new_primal_dims = list(copy.deepcopy(post.primal_dims))
#             for dim in new_dims:
#                 val_dim = sum([1 for d in new_out_dims if d.val_dim is not None])
#                 val_dim += sum([1 for d in new_primal_dims[:dim] if d.val_dim is not None and type(d) is DenseDimension])
#                 new_primal_dims.insert(dim, DenseDimension(dim, 1, val_dim))
#                 for d in new_primal_dims[dim:]:
#                     d.id += 1
#                     if d.val_dim is not None:
#                         d.val_dim += 1
#                     if type(d) is SparseDimension:
#                         _d = new_out_dims[d.other_id]
#                         _d.other_id += 1
#                         if _d.val_dim is not None:
#                             _d.val_dim += 1
                
#             new_val = jnp.expand_dims(post.val, axis=new_dims)
#             post = SparseTensor(new_out_dims, new_primal_dims, new_val, [])
#             if pre.val is None:
#                 return post
#             else:
#                 return post*pre
#     return val_out, [SparseTensor([], [], None, [concatenate_transform]) for p in primals]

# elemental_rules[lax.concatenate_p] = concatenate_elemental_rule

    
def convert_element_type_rule(primals, **params):
    # TODO check if this is actually correct
    val_out = lax.convert_element_type_p.bind(*primals, **params)

    def convert_element_type_transform(pre, post, iota):
        if post.val is None:
            pre.jac_transform = [convert_element_type_transform]
            return pre
        else:
            if pre.val is None:
                return post
            else:
                return post*pre
    return val_out, [SparseTensor([], [], None, [convert_element_type_transform])]

    
elemental_rules[lax.convert_element_type_p] = convert_element_type_rule

