import jax.lax as lax

from .base import elemental_rules, elemental_only_rules, get_shape
from ..sparse.tensor import (
    DenseDimension,
    SparseDimension,
    SparseTensor,
    _swap_back_axes,
)


def _dot_general_elementals(primals, out_shape, **params):
    """Build lhs_tensor and rhs_tensor given already-known output shape."""
    lhs, rhs = primals

    # Which dimensions of the tensors are contracted
    dimension_numbers = params["dimension_numbers"][0]
    batch_dims = params["dimension_numbers"][1]
    # NOTE: Batch dimensions are just treated as SparseDimensions.

    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]

    lhs_batch_dims = batch_dims[0]
    rhs_batch_dims = batch_dims[1]

    lhs_shape = list(get_shape(lhs))
    rhs_shape = list(get_shape(rhs))

    lhs_out_dims, rhs_out_dims = [], []
    lhs_primal_dims, rhs_primal_dims = [], []

    num_out_dims = len(out_shape)

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

                lhs_out_dims.insert(
                    batch_dim_counter,
                    SparseDimension(batch_dim_counter, ld, dim, other_lid)
                )
                lhs_primal_dims.append(
                    SparseDimension(other_lid, ld, dim, batch_dim_counter)
                )
                batch_dim_counter += 1
                for d in lhs_out_dims[batch_dim_counter:]:
                    d.id += 1
                    if isinstance(d, SparseDimension):
                        _d = lhs_primal_dims[d.other_id - num_out_dims]
                        _d.other_id += 1
            else:
                # Otherwise, we can just set `val_dim` to None
                _lid = len(lhs_out_dims)
                lhs_out_dims.append(SparseDimension(_lid, ld, None, other_lid))
                lhs_primal_dims.append(SparseDimension(other_lid, ld, None, _lid))
                rhs_out_dims.append(DenseDimension(len(rhs_out_dims), ld, lid))

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
                # If it is a batch dimension, we need to treat it as a
                # SparseDimension with a valid `val_dim`
                dim = lhs_batch_dims[jj]
                jj += 1
                rhs_out_dims.insert(
                    batch_dim_counter,
                    SparseDimension(batch_dim_counter, rd, dim, other_rid)
                )
                rhs_primal_dims.append(
                    SparseDimension(other_rid, rd, dim, batch_dim_counter)
                )
                batch_dim_counter += 1
                for d in rhs_out_dims[batch_dim_counter:]:
                    d.id += 1
                    if isinstance(d, SparseDimension):
                        _d = rhs_primal_dims[d.other_id - num_out_dims]
                        _d.other_id += 1
            else:
                # Otherwise, we can just set `val_dim` to None
                _rid = len(rhs_out_dims)
                rhs_out_dims.append(SparseDimension(_rid, rd, None, other_rid))
                rhs_primal_dims.append(SparseDimension(other_rid, rd, None, _rid))
                lhs_out_dims.append(DenseDimension(len(lhs_out_dims), rd, rid))

    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)

    lhs_tensor = _swap_back_axes(lhs_tensor)
    rhs_tensor = _swap_back_axes(rhs_tensor)
    return [lhs_tensor, rhs_tensor]


def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    out_shape = list(get_shape(val_out))
    return val_out, _dot_general_elementals(primals, out_shape, **params)


def dot_general_elemental_only(primal_out, primals, **params):
    out_shape = list(get_shape(primal_out))
    return _dot_general_elementals(primals, out_shape, **params)


elemental_rules[lax.dot_general_p] = dot_general_elemental_rule
elemental_only_rules[lax.dot_general_p] = dot_general_elemental_only
