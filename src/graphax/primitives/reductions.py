
import jax.lax as lax
import jax.numpy as jnp

from .base import elemental_rules, elemental_only_rules, get_ndim, get_shape
from ..sparse.tensor import (
    DenseDimension,
    SparseDimension,
    SparseTensor,
    _swap_back_axes,
)


# ---------- select_n ----------

def _select_elementals(primals, **params):
    size = primals[0].size
    jacsize = (size, size)
    num_cases = len(primals) - 1
    new_out_dims = [SparseDimension(0, 1, size, 1)]
    new_primal_dims = [SparseDimension(1, 1, size, 0)]
    jacval = jnp.zeros(jacsize)
    return [SparseTensor(new_out_dims, new_primal_dims, jacval) for _ in range(num_cases)]


def select_elemental_rule(primals, **params):
    val_out = lax.select_n_p.bind(*primals, **params)
    return val_out, _select_elementals(primals, **params)


def select_elemental_only(primal_out, primals, **params):
    return _select_elementals(primals, **params)


elemental_rules[lax.select_n_p] = select_elemental_rule
elemental_only_rules[lax.select_n_p] = select_elemental_only


# ---------- reduce_sum ----------

# TODO Create a general reduce rule with a custom derivative!
def _reduce_sum_elementals(primals, val_out_ndim, **params):
    primal = primals[0]
    axes = params["axes"]
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    new_out_dims, new_primal_dims, shape = [], [], []
    l = val_out_ndim  # TODO rename l, bad name...
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            # idx = len(new_out_dims) + len(new_primal_dims)
            # idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(l + i, size, count))
            shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, None, l + i))
            new_primal_dims.append(SparseDimension(l + i, size, None, ll))

    val = jnp.ones(shape, dtype=jnp.float32)
    return [SparseTensor(new_out_dims, new_primal_dims, val)]


def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)
    return val_out, _reduce_sum_elementals(primals, get_ndim(val_out), **params)


def reduce_sum_elemental_only(primal_out, primals, **params):
    return _reduce_sum_elementals(primals, get_ndim(primal_out), **params)


elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule
elemental_only_rules[lax.reduce_sum_p] = reduce_sum_elemental_only


# ---------- reduce_max ----------

def _reduce_max_elementals(primals, val_out, **params):
    primal = primals[0]
    axes = params["axes"]
    shape = list(get_shape(val_out))

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)  # TODO rename l, bad name ...
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            shape.insert(i, 1)
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(idx, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, i, l + i))
            new_primal_dims.append(SparseDimension(l + i, size, i, ll))

    _val_out = val_out.reshape(shape)
    new_val = jnp.where(primal == _val_out, 1, 0)
    # NOTE: Normalization is important if the maximum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm

    return [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]


def reduce_max_elemental_rule(primals, **params):
    val_out = lax.reduce_max_p.bind(*primals, **params)
    return val_out, _reduce_max_elementals(primals, val_out, **params)


def reduce_max_elemental_only(primal_out, primals, **params):
    return _reduce_max_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_max_p] = reduce_max_elemental_rule
elemental_only_rules[lax.reduce_max_p] = reduce_max_elemental_only


# ---------- reduce_min ----------

def _reduce_min_elementals(primals, val_out, **params):
    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    count = 0
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            idx = len(new_out_dims) + len(new_primal_dims)
            idx = max(idx, 1) if val_out.ndim > 0 else idx
            new_primal_dims.append(DenseDimension(idx, size, i))
            _shape.append(size)
            count += 1
        else:
            ll = len(new_out_dims)
            new_out_dims.append(SparseDimension(ll, size, i, l + i))
            new_primal_dims.append(SparseDimension(l + i, size, i, ll))

    new_val = jnp.where(primal == val_out, 1, 0)
    # NOTE: Normalization is important if the minimum is not unique
    norm = jnp.sum(new_val, axis=axes, keepdims=True)
    new_val = new_val / norm
    return [_swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))]


def reduce_min_elemental_rule(primals, **params):
    val_out = lax.reduce_min_p.bind(*primals, **params)
    return val_out, _reduce_min_elementals(primals, val_out, **params)


def reduce_min_elemental_only(primal_out, primals, **params):
    return _reduce_min_elementals(primals, primal_out, **params)


elemental_rules[lax.reduce_min_p] = reduce_min_elemental_rule
elemental_only_rules[lax.reduce_min_p] = reduce_min_elemental_only


# ---------- unified reduce (draft, untested) ----------

# first draft unified reduce, TODO: test!
def reduce_elemental_rule(primals, agg, **params):
    assert agg in {"sum", "min", "max"}, (
        f"{agg} is not one of the valid aggregate functions `sum`, `min`, `max`"
    )
    val_out = getattr(lax, f"reduce_{agg}_p").bind(*primals, **params)

    shape = list(get_shape(val_out))
    primal = primals[0]
    axes = params["axes"]

    new_out_dims, new_primal_dims, _shape = [], [], []
    if axes is None:
        axes = tuple(range(primal.ndim))
        new_out_dims.append(DenseDimension(0, 1, 0))
    elif isinstance(axes, int):
        axes = (axes,)

    l = get_ndim(val_out)
    for i, size in enumerate(get_shape(primal)):
        if i in axes:
            if agg == "sum":
                idx = l + i
            else:
                shape.insert(i, 1)
                idx = len(new_out_dims) + len(new_primal_dims)
                idx = max(idx, 1) if val_out.ndim > 0 else idx

            new_primal_dims.append(DenseDimension(idx, size, i))
            _shape.append(size)
        else:
            ll = len(new_out_dims)
            val = None if "sum" else i
            new_out_dims.append(SparseDimension(ll, size, val, l + i))
            new_primal_dims.append(SparseDimension(l + i, size, val, ll))

    if agg == "sum":
        new_val = jnp.ones(_shape, dtype=jnp.float32)
    else:
        _val_out = val_out.reshape(shape)
        new_val = jnp.where(primal == _val_out, 1, 0)
        norm = jnp.sum(new_val, axis=axes, keepdims=True)
        new_val /= norm

    return val_out, [
        _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))
    ]


# elemental_rules[lax.reduce_sum_p] = partial(reduce_elemental_rule, agg="sum")
# elemental_rules[lax.reduce_min_p] = partial(reduce_elemental_rule, agg="min")
# elemental_rules[lax.reduce_max_p] = partial(reduce_elemental_rule, agg="max")
