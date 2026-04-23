import inspect
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax._src.core as core

from ..sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
    _swap_back_axes,
)


def get_ndim(arr):
    if isinstance(arr, (float, int, jax._src.literals.TypedFloat)):
        return 0
    else:
        return arr.ndim


def get_shape(arr):
    if isinstance(arr, (float, int, jax._src.literals.TypedFloat)):
        return ()
    else:
        return arr.shape


def get_aval_shape(val):
    if isinstance(val, np.ndarray):
        return val.shape
    else:
        return ()


def make_parallel_jacobian(i, primals, val_out, elemental):
    if len(primals) > 2:
        raise NotImplementedError(f"Parallel Jacobians with {len(primals)} inputs not yet supported!")

    primal = primals[i]
    primal_size = get_ndim(primal)
    out_size = get_ndim(val_out)
    out_shape = get_shape(val_out)

    if primal_size == 0 and out_size == 0:
        # Singletons
        return SparseTensor([], [], elemental)

    if primal_size == 0:
        # Broadcast singleton
        elemental_is_scalar = (
            type(elemental) is float
            or (hasattr(elemental, 'size') and elemental.size == 1)
        )
        if elemental_is_scalar:
            if type(elemental) is not float:
                elemental = jnp.squeeze(elemental)
            out_dims = [DenseIndex(j, e, None) for j, e in enumerate(val_out.aval.shape)]
        else:
            out_dims = [DenseIndex(j, e, j) for j, e in enumerate(val_out.aval.shape)]
        return SparseTensor(out_dims, [], elemental)

    if len(primals) == 2 and get_shape(primal) != get_shape(val_out):
        # Broadcasting case
        elemental_is_scalar = (
            type(elemental) is float
            or (hasattr(elemental, 'size') and elemental.size == 1)
        )
        if elemental_is_scalar and type(elemental) is not float:
            elemental = jnp.squeeze(elemental)
        out_dims, primal_dims = [], []
        for j, (os, ps) in enumerate(zip(val_out.aval.shape, primal.aval.shape)):
            n_out, n_primal = len(out_dims), len(primal_dims)
            if ps != os:
                val_axis = None if elemental_is_scalar else sum(1 for d in out_dims if d.val_axis is not None)
                out_dims.append(DenseIndex(j, os, val_axis))
                primal_dims.append(
                    DenseIndex(n_out + n_primal + 1, ps, None)
                )
            else:
                val_axis = None if elemental_is_scalar else sum(1 for d in out_dims if d.size is not None)
                out_dims.append(
                    SparseIndex(j, os, val_axis, n_out + n_primal + 1)
                )
                primal_dims.append(
                    SparseIndex(n_out + n_primal + 1, os, val_axis, j)
                )
            for d in primal_dims[:-1]:
                d.id += 1
                if isinstance(d, SparseIndex):
                    out_dims[d.other_id].other_id += 1
        return _swap_back_axes(SparseTensor(out_dims, primal_dims, elemental))

    if type(elemental) is float or (hasattr(elemental, 'size') and elemental.size == 1):
        if type(elemental) is not float:
            elemental = jnp.squeeze(elemental)
        val_axis_fn = lambda _: None
    else:
        val_axis_fn = lambda j: j

    shape = primal.aval.shape
    out_dims = [SparseIndex(j, e, val_axis_fn(j), out_size + j)
                for j, e in enumerate(shape)]
    primal_dims = [SparseIndex(out_size + j, e, val_axis_fn(j), j)
                   for j, e in enumerate(shape)]
    return SparseTensor(out_dims, primal_dims, elemental)


elemental_rules = {}
# Maps primitive -> (primal_out, primals, **params) -> list[SparseTensor]
# Elemental computation only; primal_out is passed in so primitive.bind is
# never called inside the thunk.
elemental_only_rules = {}
# Maps primitive -> (primal_outs, primals, **params) -> list[list[SparseTensor]]
# For primitives with multiple_results=True. Returns elementals[outvar_idx][invar_idx].
multi_output_elemental_only_rules = {}


def _filter_params(fn, params):
    """Filter params to only those accepted by fn, to handle new JAX params gracefully."""
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return params
        valid = set(sig.parameters.keys())
        return {k: v for k, v in params.items() if k in valid}
    except (ValueError, TypeError):
        return params


def standard_elemental(elementalrule, primitive, primals, **params):
    assert elementalrule is not None, f"Elemental rule does exist for {primitive}!"
    val_out = primitive.bind(*primals, **params)

    elementals = elementalrule(*primals, **_filter_params(elementalrule, params))
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)

    elementals_out = [
        make_parallel_jacobian(i, primals, val_out, elemental)
        for i, elemental in enumerate(elementals)
        if not isinstance(primals[i], (float, np.ndarray, np.float32))
    ]
    return val_out, elementals_out


def standard_elemental_only(elementalrule, primal_out, primals, **params):
    """Elemental-only variant: takes already-computed primal, never calls primitive.bind."""
    _filtered_params = _filter_params(elementalrule, params)
    elementals = elementalrule(*primals, **_filtered_params)
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)
    return [
        make_parallel_jacobian(i, primals, primal_out, elemental)
        for i, elemental in enumerate(elementals)
        if not isinstance(primals[i], (float, np.ndarray, np.float32))
    ]


def defelemental(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental, elementalrule, primitive)
    elemental_only_rules[primitive] = partial(standard_elemental_only, elementalrule)


def standard_elemental2(elementalrule, primitive, primals, **params):
    assert elementalrule is not None

    val_out = primitive.bind(*primals, **params)
    _filtered_params = _filter_params(elementalrule, params)

    elementals = elementalrule(val_out, *primals, **_filtered_params)
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)
    elementals_out = [
        make_parallel_jacobian(i, primals, val_out, elemental)
        for i, elemental in enumerate(elementals)
        if not isinstance(primals[i], (float, np.ndarray, np.float32))
    ]
    return val_out, elementals_out


def standard_elemental2_only(elementalrule, primal_out, primals, **params):
    """Elemental-only variant for output-dependent rules: primal_out passed in."""
    _filtered_params = _filter_params(elementalrule, params)
    elementals = elementalrule(primal_out, *primals, **_filtered_params)
    elementals = elementals if isinstance(elementals, tuple) else (elementals,)
    return [
        make_parallel_jacobian(i, primals, primal_out, elemental)
        for i, elemental in enumerate(elementals)
        if not isinstance(primals[i], (float, np.ndarray, np.float32))
    ]


def defelemental2(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental2, elementalrule, primitive)
    elemental_only_rules[primitive] = partial(standard_elemental2_only, elementalrule)
