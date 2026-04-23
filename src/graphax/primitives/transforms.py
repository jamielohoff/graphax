import copy

from typing import Callable
from functools import partial

import jax.lax as lax
import jax.numpy as jnp

from .base import elemental_rules, elemental_only_rules
from ..sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
    _materialize_axes,
    _swap_back_axes,
)

Transform = Callable[[SparseTensor, SparseTensor, jnp.ndarray], SparseTensor]


class JacobianTransform:
    transform: Transform
    inverse_transform: Transform

    def __init__(
        self, transform: Transform, inverse_transform: Transform = None
    ) -> None:
        self.transform = transform
        self.inverse_transform = inverse_transform

    def __repr__(self) -> str:
        return (
            f"JacobianTransform(transform={self.transform}, "
            f"inverse_transform={self.inverse_transform})"
        )

    def apply(self, tensor: SparseTensor, iota: jnp.ndarray) -> SparseTensor:
        if self.transform is None:
            raise NotImplementedError("Transform not implemented!")
        return self.transform(tensor, iota)

    def apply_inverse(self, tensor: SparseTensor, iota: jnp.ndarray) -> SparseTensor:
        if self.inverse_transform is None:
            raise NotImplementedError("Inverse transform not implemented!")
        return self.inverse_transform(tensor, iota)


def _inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for i, p in enumerate(permutation):
        inverse[p] = i
    return inverse


# ---------- transpose ----------

def _transpose_elementals(primals, val_out, **params):
    permutation = params["permutation"]

    def transpose_transform(pre, iota):
        new_out_dims = []
        new_primal_dims = pre.primal_dims
        counter = 0
        l = len(pre.out_dims)

        for p in permutation:
            new_out_dims.append(pre.out_dims[p])
            new_out_dims[-1].id = counter
            if isinstance(new_out_dims[-1], SparseIndex):
                other_id = new_out_dims[-1].other_id
                new_primal_dims[other_id - l].other_id = counter
            counter += 1

        return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, pre.val))

        return _swap_back_axes(
            SparseTensor(new_out_dims, new_primal_dims, pre.val)
        )

    def inverse_transpose_transform(post, iota):
        new_out_dims = post.out_dims
        new_primal_dims = []
        counter = len(post.out_dims)

        # This implementation is faulty!
        inv_permutation = _inverse_permutation(permutation)
        for p in inv_permutation:
            new_primal_dims.append(post.primal_dims[p])
            new_primal_dims[-1].id = counter
            if isinstance(new_primal_dims[-1], SparseIndex):
                other_id = new_primal_dims[-1].other_id
                new_out_dims[other_id].other_id = counter
            counter += 1

        return _swap_back_axes(
            SparseTensor(new_out_dims, new_primal_dims, post.val)
        )

    transform = JacobianTransform(transpose_transform, inverse_transpose_transform)
    return [SparseTensor([], [], None, [transform])]


# Should work for high-dimensional stuff
def transpose_elemental_rule(primals, **params):
    val_out = lax.transpose_p.bind(*primals, **params)
    return val_out, _transpose_elementals(primals, val_out, **params)


def transpose_elemental_only(primal_out, primals, **params):
    return _transpose_elementals(primals, primal_out, **params)


elemental_rules[lax.transpose_p] = transpose_elemental_rule
elemental_only_rules[lax.transpose_p] = transpose_elemental_only


# ---------- reshape ----------

def _reshape_elementals(primals, val_out, **params):
    # TODO: dimensional collapse is not covered here!
    # Implement sparsity-aware version for significant speedup!

    def reshape_transform(pre, iota):
        # NOTE array is not correctly materialized sometimes!
        full_val = pre.dense(iota)
        new_shape = []
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
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
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        for s in primals[0].shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1
        full_val = full_val.reshape(new_shape)
        return SparseTensor(new_out_dims, new_primal_dims, full_val)

    transform = JacobianTransform(reshape_transform, inverse_reshape_transform)
    return [SparseTensor([], [], None, [transform])]


def reshape_elemental_rule(primals, **params):
    val_out = lax.reshape_p.bind(*primals, **params)
    return val_out, _reshape_elementals(primals, val_out, **params)


def reshape_elemental_only(primal_out, primals, **params):
    return _reshape_elementals(primals, primal_out, **params)


elemental_rules[lax.reshape_p] = reshape_elemental_rule
elemental_only_rules[lax.reshape_p] = reshape_elemental_only


# ---------- slice ----------

def _slice_elementals(primals, val_out, **params):
    # The slice primitive is written in such a way that it just densifies the
    # Jacobian and then slices it. This is not efficient and there might be ways
    # to make this more efficient by checking if sparse dimensions are untouched
    # how this changes the Jacobian.

    def slice_transform(pre, iota):
        start_indices = list(params["start_indices"])
        limit_indices = list(params["limit_indices"])
        full_val = pre.dense(iota)
        new_out_dims = []
        new_primal_dims = []
        counter = 0

        for s in val_out.shape:
            new_out_dims.append(DenseIndex(counter, s, counter))
            counter += 1

        for d in pre.primal_dims:
            new_primal_dims.append(DenseIndex(counter, d.size, counter))
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
            new_out_dims.append(DenseIndex(counter, d.size, counter))
            new_shape.append(d.size)
            counter += 1
        scatter_zeros = jnp.zeros(counter, dtype=jnp.int32)

        for s in primals[0].shape:
            new_primal_dims.append(DenseIndex(counter, s, counter))
            new_shape.append(s)
            counter += 1

        zeros = jnp.zeros(new_shape)
        dims = tuple(range(zeros.ndim))
        scatter_dims = lax.ScatterIndexNumbers(dims, (), dims)
        _scatter_indices = jnp.array(start_indices, dtype=jnp.int32)
        scatter_indices = jnp.concatenate([scatter_zeros, _scatter_indices])

        new_val = lax.scatter(zeros, scatter_indices, full_val, scatter_dims)

        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    transform = JacobianTransform(slice_transform, inverse_slice_transform)
    return [SparseTensor([], [], None, [transform])]


def slice_elemental_rule(primals, **params):
    val_out = lax.slice_p.bind(*primals, **params)
    return val_out, _slice_elementals(primals, val_out, **params)


def slice_elemental_only(primal_out, primals, **params):
    return _slice_elementals(primals, primal_out, **params)


elemental_rules[lax.slice_p] = slice_elemental_rule
elemental_only_rules[lax.slice_p] = slice_elemental_only


# ---------- broadcast_in_dim ----------

def _broadcast_elementals(primals, val_out, **params):
    # Materialize the broadcast Jacobian as a SparseTensor instead of a deferred
    # JacobianTransform. The Jacobian is the Kronecker delta on matched axes and
    # constant (1.0) along broadcasted axes, so this costs only dimension
    # metadata — no tensor data is allocated.
    #
    # The previous transform-based approach worked in forward mode (transforms
    # got consumed during composition with a val-carrying pre), but broke in
    # reverse mode: when a transform-only edge was composed as `pre` with a
    # val-carrying `post`, the transform was appended to the resulting edge's
    # pre_transforms. Later attempts to apply it to that composed edge mutated
    # out_dims (the transform's design), but the broadcast actually needed to
    # be resolved on the primal side of the composed edge.
    dims = params["broadcast_dimensions"]
    shape = params["shape"]
    primal = primals[0]
    primal_shape = primal.shape if hasattr(primal, "shape") else ()

    l = len(shape)
    n = len(primal_shape)
    # dims[j] is the output axis where primal axis j is placed; any remaining
    # output axes are new broadcast axes.
    new_out_dims = []
    for i in range(l):
        if i in dims:
            primal_idx = dims.index(i)
            new_out_dims.append(SparseIndex(i, shape[i], None, l + primal_idx))
        else:
            new_out_dims.append(DenseIndex(i, shape[i], None))
    new_primal_dims = [
            SparseIndex(l + j, primal_shape[j], None, dims[j])
            for j in range(n)
        ]
    return [SparseTensor(new_out_dims, new_primal_dims, 1.0)]

def broadcast_elemental_rule(primals, **params):
    val_out = lax.broadcast_in_dim_p.bind(*primals, **params)
    return val_out, _broadcast_elementals(primals, val_out, **params)


def broadcast_elemental_only(primal_out, primals, **params):
    return _broadcast_elementals(primals, primal_out, **params)


elemental_rules[lax.broadcast_in_dim_p] = broadcast_elemental_rule
elemental_only_rules[lax.broadcast_in_dim_p] = broadcast_elemental_only


# ---------- squeeze ----------

def _squeeze_elementals(primals, val_out, **params):
    # NOTE: squeeze is basically just the inverse operation to broadcast_in_dim
    # since it just adds a DenseIndex of size 1

    def squeeze_transform(pre, iota):
        dims = sorted(params["dimensions"])
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        squeeze_dims = []
        counter = 0

        for id in dims:
            idx = [j for j, d in enumerate(new_out_dims) if d.id == id][0]
            val_axis = new_out_dims[idx].val_axis
            squeeze_dims.append(val_axis)

            if isinstance(new_out_dims[idx], SparseIndex):
                def _check(d, id):
                    if isinstance(d, SparseIndex):
                        return d.other_id == id
                    else:
                        return False
                other_idx = [j for j, d in enumerate(new_primal_dims)
                             if _check(d, id)][0]
                other_dim = new_primal_dims[other_idx]
                new_primal_dims[other_idx] = DenseIndex(
                    other_dim.id, other_dim.size, None
                )

            del new_out_dims[idx]
            counter += 1

        out_ids = [d.id for d in new_out_dims]
        primal_ids = [d.id for d in new_primal_dims]
        new_val_axiss = [d.val_axis for d in new_out_dims
                        if d.val_axis is not None]
        new_val_axiss += [d.val_axis for d in new_primal_dims
                         if isinstance(d, DenseIndex) and d.val_axis is not None]

        for d in new_out_dims:
            d.id = out_ids.index(d.id)
            if d.val_axis is not None:
                d.val_axis = new_val_axiss.index(d.val_axis)
            if isinstance(d, SparseIndex):
                d.other_id = len(new_out_dims) + primal_ids.index(d.other_id)

        for d in new_primal_dims:
            d.id = len(new_out_dims) + primal_ids.index(d.id)
            if d.val_axis is not None:
                d.val_axis = new_val_axiss.index(d.val_axis)
            if isinstance(d, SparseIndex):
                d.other_id = out_ids.index(d.other_id)

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
            val_axis = sum(1 for d in new_out_dims if d.val_axis is not None)
            val_axis += sum(1 for d in new_primal_dims[:dim]
                           if d.val_axis is not None and isinstance(d, DenseIndex))
            new_primal_dims.insert(dim, DenseIndex(dim, 1, val_axis))
            for d in new_primal_dims[dim:]:
                d.id += 1
                if d.val_axis is not None:
                    d.val_axis += 1
                if isinstance(d, SparseIndex):
                    _d = new_out_dims[d.other_id]
                    _d.other_id += 1
                    if _d.val_axis is not None:
                        _d.val_axis += 1

        new_val = jnp.expand_dims(post.val, axis=new_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    transform = JacobianTransform(squeeze_transform, inverse_squeeze_transform)
    return [SparseTensor([], [], None, [transform])]


def squeeze_elemental_rule(primals, **params):
    val_out = lax.squeeze_p.bind(*primals, **params)
    return val_out, _squeeze_elementals(primals, val_out, **params)


def squeeze_elemental_only(primal_out, primals, **params):
    return _squeeze_elementals(primals, primal_out, **params)


elemental_rules[lax.squeeze_p] = squeeze_elemental_rule
elemental_only_rules[lax.squeeze_p] = squeeze_elemental_only


# ---------- concatenate ----------

def _concatenate_elementals(primals, val_out, **params):
    # This gradient transformation is designed to take an post edge and
    # decompose it into the pre edges. This is done by densifying the post along
    # the respective axes and then use jnp.split to split the tensor.
    # TODO DynamicJaxprTracer is now a unhashable type, so we can no longer use
    # it as a key in the dict. We need to find another way of doing this.
    dim = params["dimension"]

    offset = primals[0].shape[dim]
    slices = {0: [0, offset]}
    for i, val in enumerate(primals[1:], start=1):
        slices[i] = [offset, offset + val.shape[dim]]
        offset += val.shape[dim]

    def concatenate_transform(primal, pre, iota):
        new_out_dims = list(copy.deepcopy(pre.out_dims))
        new_primal_dims = list(copy.deepcopy(pre.primal_dims))
        l = len(pre.out_dims)

        d = new_out_dims[dim]
        dim_id = d.id
        primal_idx = [idx for idx, p in enumerate(primals) if p is primal][0]
        idx, _idx = slices[primal_idx]

        if isinstance(d, DenseIndex):
            if d.val_axis is not None:
                lshape = list(pre.val.shape)
                rshape = list(pre.val.shape)
                lshape[d.val_axis] = idx
                rshape[d.val_axis] = val_out.shape[dim] - _idx
                lcat_zeros = jnp.zeros(lshape)
                rcat_zeros = jnp.zeros(rshape)

                new_val = jnp.concatenate(
                    [lcat_zeros, pre.val, rcat_zeros], axis=d.val_axis
                )

                new_out_dims[dim].size = new_val.shape[d.val_axis]
            else:
                # val_axis=None: this output dimension is a Kronecker factor not stored in val.
                # Materialize it: broadcast pre.val to the primal's slice size (zero-copy),
                # then pad with zeros in a single lax.pad pass.
                val_size = _idx - idx
                new_val_axis = sum(1 for dd in new_out_dims[:dim]
                                  if dd.val_axis is not None)

                new_val = jnp.expand_dims(pre.val, axis=new_val_axis)
                new_val = jnp.broadcast_to(
                    new_val,
                    (*pre.val.shape[:new_val_axis], val_size, *pre.val.shape[new_val_axis:])
                )

                pad_config = [(0, 0, 0)] * new_val.ndim
                pad_config[new_val_axis] = (idx, val_out.shape[dim] - _idx, 0)
                new_val = lax.pad(
                    new_val, jnp.zeros((), dtype=new_val.dtype), pad_config
                )

                # Inserting a new axis shifts all subsequent val_axiss up by 1
                for _dim in new_out_dims[dim + 1:]:
                    if _dim.val_axis is not None:
                        _dim.val_axis += 1
                for _dim in new_primal_dims:
                    if isinstance(_dim, DenseIndex) and _dim.val_axis is not None:
                        _dim.val_axis += 1

                new_out_dims[dim].val_axis = new_val_axis
                new_out_dims[dim].size = val_out.shape[dim]
        else:
            other_id = d.other_id
            if d.val_axis is not None:
                _d = new_primal_dims[other_id - l]

                # Calculate the new val_axis of the primal dimension
                val_axis = sum(1 for dd in new_out_dims if dd.val_axis is not None)
                val_axis += sum(1 for dd in new_primal_dims[:other_id - l]
                               if dd.val_axis is not None and isinstance(dd, DenseIndex))

                # Update the val_axis of all following dimensions
                for _dim in new_primal_dims[dim + 1:]:
                    if isinstance(_dim, DenseIndex) and _dim.val_axis is not None:
                        _dim.val_axis += 1

                # Materialize the sparse dimensions related to the concatenation dimension
                new_val = _materialize_axes(pre, [d.id])

                if iota.shape[0] < d.size or iota.shape[1] < d.size:
                    sub_iota = jnp.eye(d.size, dtype=jnp.float32)
                else:
                    sub_iota = lax.slice(iota, [0, 0], [d.size, d.size])

                shape = [1 for _ in range(pre.val.ndim)]
                shape[_d.val_axis] = _d.size
                shape.insert(val_axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                # Make zeros for insertion
                _size = val_out.shape[dim]
                _shape = list(new_val.shape)
                _shape[d.val_axis] = _size
                _shape[val_axis] = d.size
                zeros = jnp.zeros(_shape, dtype=jnp.float32)

                # scatter_indices: where in `zeros` to place `new_val`
                scatter_indices = [0 for _ in _shape]
                scatter_indices[d.val_axis] = idx
                scatter_indices[val_axis] = 0

                update_window_dims = tuple(range(len(_shape)))
                scatter_dims_to_operand_dims = tuple(range(len(_shape)))

                scatter_dims = lax.ScatterIndexNumbers(
                    update_window_dims, (), scatter_dims_to_operand_dims
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array(scatter_indices),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True
                )

                new_out_dims[dim_id] = DenseIndex(
                    dim_id, val_out.shape[dim], d.val_axis
                )
                new_primal_dims[other_id - l] = DenseIndex(
                    other_id, d.size, val_axis
                )
            else:
                _d = new_primal_dims[other_id - l]
                _size = val_out.shape[dim]

                # Calculate the new val_axis of the out dimension
                out_val_axis = sum(1 for dd in new_out_dims[:dim]
                                  if dd.val_axis is not None)

                # Calculate the new val_axis of the primal dimension
                primal_val_axis = sum(1 for dd in new_out_dims if dd.val_axis is not None)
                primal_val_axis += sum(1 for dd in new_primal_dims[:other_id - l]
                                      if dd.val_axis is not None and isinstance(dd, DenseIndex))
                primal_val_axis = max(1, primal_val_axis)

                # Update the val_axis of all following dimensions
                for _dim in new_primal_dims[dim + 1:]:
                    if isinstance(_dim, DenseIndex) and _dim.val_axis is not None:
                        _dim.val_axis += 1

                # Materialize the sparse dimensions related to the concatenation dimension
                if pre.val.shape != ():
                    new_val = _materialize_axes(pre, [d.id, d.other_id])
                else:
                    new_val = pre.val

                if iota.shape[0] < d.size or iota.shape[1] < d.size:
                    sub_iota = jnp.eye(d.size, dtype=jnp.float32)
                else:
                    sub_iota = lax.slice(iota, [0, 0], [d.size, d.size])

                shape = [1 for _ in range(pre.val.ndim)]
                shape.insert(out_val_axis, _d.size)
                shape.insert(primal_val_axis, d.size)

                new_val = new_val * sub_iota

                # Make zeros for insertion
                _shape = list(pre.val.shape)
                _shape.insert(out_val_axis, _size)
                _shape.insert(primal_val_axis, _d.size)
                zeros = jnp.zeros(_shape, dtype=jnp.float32)

                scatter_dims = lax.ScatterIndexNumbers(
                    (out_val_axis, primal_val_axis), (), (out_val_axis, primal_val_axis)
                )
                new_val = lax.scatter(
                    zeros,
                    jnp.array([idx, 0]),
                    new_val,
                    scatter_dims,
                    indices_are_sorted=True,
                    unique_indices=True
                )

                new_out_dims[dim_id] = DenseIndex(
                    dim_id, val_out.shape[dim], out_val_axis
                )
                new_primal_dims[other_id - l] = DenseIndex(
                    other_id, d.size, primal_val_axis
                )

        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    def inverse_concatenate_transform(primal, post, iota):
        new_out_dims = list(copy.deepcopy(post.out_dims))
        new_primal_dims = list(copy.deepcopy(post.primal_dims))
        primal_idx = next(idx for idx, p in enumerate(primals) if p is primal)

        d = None
        if len(new_primal_dims) > 0:
            d = new_primal_dims[dim]
        if d is None:
            # post is a pure transform with no primal dims; nothing to slice.
            new_val = post.val
        elif isinstance(d, DenseIndex):
            if d.val_axis is not None:
                new_val = lax.slice_in_dim(
                    post.val, *slices[primal_idx], axis=d.val_axis
                )
                d.size = new_val.shape[d.val_axis]
            else:
                # val_axis=None: the primal dimension is a Kronecker factor not stored in val.
                # There is no axis to slice — just narrow the size to this primal's contribution.
                d.size = slices[primal_idx][1] - slices[primal_idx][0]
                new_val = post.val
        else:
            _d = new_out_dims[d.other_id]
            if d.val_axis is not None:
                new_out_dims[d.other_id] = DenseIndex(_d.id, _d.size, _d.val_axis)
                size = slices[primal_idx][1] - slices[primal_idx][0]

                # Calculate the new val_axis of the primal dimension
                val_axis = sum(1 for dd in new_out_dims if dd.val_axis is not None)
                val_axis += sum(1 for dd in new_primal_dims[:dim]
                               if dd.val_axis is not None and type(dd) is DenseIndex)
                new_primal_dims[dim] = DenseIndex(_d.other_id, size, val_axis)

                # Update the val_axis of all following dimensions
                for _dim in new_primal_dims[dim + 1:]:
                    if type(_dim) is DenseIndex and _dim.val_axis is not None:
                        _dim.val_axis += 1

                # Materialize the sparse dimensions related to the concatenation dimension
                new_val = _materialize_axes(post, [d.id])

                if iota.shape[0] < d.size or iota.shape[1] < d.size:
                    sub_iota = jnp.eye(d.size, dtype=jnp.float32)
                else:
                    sub_iota = lax.slice(iota, [0, 0], [d.size, d.size])

                shape = [1 for _ in range(post.val.ndim)]
                shape[_d.val_axis] = _d.size
                shape.insert(val_axis, d.size)
                sub_iota = sub_iota.reshape(shape)

                new_val = new_val * sub_iota

                new_val = lax.slice_in_dim(
                    new_val, *slices[primal_idx], axis=val_axis
                )
                d.size = new_val.shape[d.val_axis]
                _d.size = new_val.shape[d.val_axis]
            else:
                # d is SparseIndex with val_axis=None:
                # Both d and its partner _d are implicit Kronecker factors not stored in val.
                # Just narrow sizes to this primal's slice; no val axis to manipulate.
                size = slices[primal_idx][1] - slices[primal_idx][0]
                d.size = size
                _d.size = size
                new_val = post.val
        return SparseTensor(new_out_dims, new_primal_dims, new_val)

    return [
        SparseTensor([], [], None,
            [
                JacobianTransform(
                    partial(concatenate_transform, p),
                    partial(inverse_concatenate_transform, p),
                )
            ],
        )
        for p in primals
    ]


def concatenate_elemental_rule(primals, **params):
    val_out = lax.concatenate_p.bind(*primals, **params)
    return val_out, _concatenate_elementals(primals, val_out, **params)


def concatenate_elemental_only(primal_out, primals, **params):
    return _concatenate_elementals(primals, primal_out, **params)


elemental_rules[lax.concatenate_p] = concatenate_elemental_rule
elemental_only_rules[lax.concatenate_p] = concatenate_elemental_only


# ---------- convert_element_type ----------

def _convert_element_type_elementals(primals, val_out, **params):
    new_dtype = params["new_dtype"]

    def convert_element_type_transform(pre, iota):
        new_pre_val = lax.convert_element_type(pre.val, new_dtype)
        new_out_dims = copy.deepcopy(pre.out_dims)
        new_primal_dims = copy.deepcopy(pre.primal_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_pre_val)

    def inverse_convert_element_type_transform(post, iota):
        new_post_val = lax.convert_element_type(post.val, new_dtype)
        new_out_dims = copy.deepcopy(post.out_dims)
        new_primal_dims = copy.deepcopy(post.primal_dims)
        return SparseTensor(new_out_dims, new_primal_dims, new_post_val)

    transform = JacobianTransform(
        convert_element_type_transform, inverse_convert_element_type_transform
    )
    return [SparseTensor([], [], None, [transform])]


def convert_element_type_rule(primals, **params):
    val_out = lax.convert_element_type_p.bind(*primals, **params)
    return val_out, _convert_element_type_elementals(primals, val_out, **params)


def convert_element_type_only(primal_out, primals, **params):
    return _convert_element_type_elementals(primals, primal_out, **params)


elemental_rules[lax.convert_element_type_p] = convert_element_type_rule
elemental_only_rules[lax.convert_element_type_p] = convert_element_type_only
