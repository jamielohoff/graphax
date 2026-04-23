"""
Sparse tensor algebra implementation
"""
import copy
from dataclasses import dataclass
from typing import Callable, Generator, Sequence, Union, Tuple
import jax.lax as lax
import jax.numpy as jnp


from jax import Array

from .utils import eye_like_copy, eye_like


# NOTE: a val_axis of None means that we have a possible replication of the tensor
#   along the respective index `d.size` times to manage broadcasting
#   operations such as broadcasted additions or multiplications.
# TODO: what do we do when we have a tensor that consists only of DenseIndices
#   with val_axis=None?
@dataclass
class DenseIndex:
    id: int
    size: int
    val_axis: int

# NOTE: a val_axis of None means that we have a factored Kronecker delta in
#   our tensor at the respective indices.
#   Also we can have unmatching `size` and `val.shape[d.val_axis]` for SparseIndices
#   if the size is 1. This is necessary to enable broadcasting operations.
@dataclass
class SparseIndex:
    id: int
    size: int
    val_axis: int
    other_id: int

Index = Union[DenseIndex, SparseIndex]
        

class SparseTensor:
    """
    The `SparseTensor object enables` the representation of sparse tensors
    that if out_dims or primal_dims is empty, this implies a scalar dependent or
    independent variable. if both are empty, then we have a scalar value and
    everything becomes trivial and the `val` field contains the value of the
    singleton partial
    """
    out_dims: Tuple[Index]
    primal_dims: Tuple[Index] # input indices
    dims: Tuple[Index]
    shape: Tuple[int] # True shape of the tensor
    val: Array
    pre_transforms: Sequence[Callable]
    post_transforms: Sequence[Callable]
    # TODO: Document pre_transforms and post_transforms. What about addition?
    # NOTE: We always assume that the indices are ordered in ascending order
    
    def __init__(self, 
                 out_dims: Sequence[Index],
                 primal_dims: Sequence[Index],
                 val: Array,
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)
        self.dims = self.out_dims + self.primal_dims

        self.out_shape = [d.size for d in out_dims]
        self.primal_shape = [d.size for d in primal_dims]

        self.shape = tuple(self.out_shape + self.primal_shape)

        self.val = jnp.asarray(val) if val is not None else None
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

        _assert_sparse_tensor_consistency(self)
            
    def __repr__(self) -> str:
        def map_str(a: Sequence) -> Generator:
            return (str(s) for s in a)

        def multiline_seq(s: Sequence, brackets: str) -> str:
            lb, rb, *_ = brackets
            if s:
                return f"{lb}\n    " + ",\n    ".join(map_str(s)) + f",\n  {rb}"
            else:
                return lb + rb

        str_out_shape = ", ".join(map_str(self.out_shape))
        str_primal_shape = ", ".join(map_str(self.primal_shape))

        multiline_out_dims = multiline_seq(self.out_dims, "()")
        multiline_primal_dims = multiline_seq(self.primal_dims, "()")
        multiline_pre_transform = multiline_seq(self.pre_transforms, "[]")
        multiline_post_transform = multiline_seq(self.post_transforms, "[]")

        return f"SparseTensor(\n" \
               f"  shape = ({str_out_shape} | {str_primal_shape}),\n"\
               f"  out_dims = {multiline_out_dims},\n"\
               f"  primal_dims = {multiline_primal_dims},\n"\
               f"  val = {self.val},\n"\
               f"  pre_transforms = {multiline_pre_transform},\n"\
               f"  post_transforms = {multiline_post_transform}\n"\
               f")"\
                    
    def __add__(self, _tensor):
        return _add(self, _tensor)
    
    def __mul__(self, _tensor):
        return _mul(self, _tensor)

    # TODO: add the case where `val_axis = None` for a `DenseIndex` by
    #   replicating the tensor `d.size` times using `jnp.tile`.
    def dense(self, iota: Array) -> Array:
        """
        Materializes tensor to actual dense shape.

        Args:
            iota (Array): The Kronecker matrix/tensor that is used to 
                materialize the tensor.

        Returns:
            Array: Dense representation of the sparse tensor.
        """
        # Compute shape of the multiindexal eye with which the `val` tensor
        # will get multiplied to manifest the sparse indices  
        # If tensor contains SparseIndices, we have to materialize them
        def eye_dim_fn(d: Index) -> int:
            if isinstance(d, SparseIndex):
                return d.size
            else:
                return 1

        # Early returns that don't require materializing the identity tensor.
        # These checks are resolved at trace time (shapes are static), so
        # performing them before eye_like_copy avoids emitting dead
        # broadcast_in_dim equations into the jaxpr.
        if self.val is not None:
            if self.val.shape == self.shape:
                return self.val
            if not self.out_dims and not self.primal_dims:
                return self.val

        eye_shape = [eye_dim_fn(d) for d in self.dims]
        eye = eye_like_copy(eye_shape, len(self.out_dims), iota)

        # If tensor consists only of Kronecker deltas, return the eye directly.
        if self.val is None:
            return eye

        shape = _get_fully_materialized_shape(self)
        val = self.val.reshape(shape) * eye
        
        # Get the tiling for DenseIndices with `val_axis = None`, i.e. replicating
        # indices
        def tile_dim_fn(d: Index) -> int:
            if isinstance(d, DenseIndex) and d.val_axis is None:
                return d.size
            else:
                return 1

        tiling = [tile_dim_fn(d) for d in self.dims]
        index_map = eye_like_copy(eye_shape, len(self.out_dims), iota)
        return jnp.tile(index_map*val, tiling)
        
    def copy(self, val: Array = None):
        """
        Function that copies the given sparse tensor object entirely except for
        the `val` property which can be replaced by a new value.

        Args:
            val (Array, optional): The new value of the `val` property. Defaults to None.

        Returns:
            SparseTensor: A copy of the original SparseTensor object.
        """
        out_dims = copy.deepcopy(self.out_dims)
        primal_dims = copy.deepcopy(self.primal_dims)
        val = self.val if val is None else val
        return SparseTensor(out_dims, primal_dims, val, self.pre_transforms, self.post_transforms)


def sparse_tensor_zeros_like(st: SparseTensor) -> SparseTensor:
    """
    Function that generates a new `SparseTensor` with only zeros with the 
    same shape as the given `SparseTensor` `st`.

    Args:
        st (SparseTensor): `SparseTensor` whose shape we want to use to initialize
                            the new `SparseTensor` object.
    Returns:
        SparseTensor:  A copy of a given `SparseTensor` object with only zeros.
    """
    return st.copy(jnp.zeros_like(st.val))
    
    
def _assert_sparse_tensor_consistency(st: SparseTensor):
    """
    Function that validates the consistency of a `SparseTensor` object,
    i.e. checks if the `val` property has the correct shape and if the indices
    are ordered correctly and sizes match the shape of `val`.

    Args:
        st (SparseTensor): SparseTensor object we want to validate.

    Returns:
        bool: True if the `SparseTensor` object is consistent.
    """
    # Check if d.size matches val.shape[d.val_axis] for all d
    matching_sparse_sizes = all(
        (d.val_axis < len(st.val.shape) and (d.size == st.val.shape[d.val_axis] or d.size == 1))
        if isinstance(d, SparseIndex) and d.val_axis is not None else True
        for d in st.dims
    )

    matching_dense_sizes = all(
        (d.val_axis < len(st.val.shape) and d.size == st.val.shape[d.val_axis])
        if isinstance(d, DenseIndex) and d.val_axis is not None else True
        for d in st.dims
    )

    matching_sizes = matching_sparse_sizes or matching_dense_sizes
        
    unique_out_dims = [d.val_axis for d in st.out_dims if d.val_axis is not None]
    unique_primal_dims = [d.val_axis for d in st.primal_dims if d.val_axis is not None]
    
    is_uniqe_out_dims = len(unique_out_dims) == len(set(unique_out_dims))
    is_uniqe_primal_dims = len(unique_primal_dims) == len(set(unique_primal_dims))
    has_uniqe_dims = is_uniqe_out_dims and is_uniqe_primal_dims

    # Check if IDs in out_dims and primal_dims match their index positions
    matching_id = all(
        od.id == i and pd.id == i + len(st.out_dims)
        for i, (od, pd) in enumerate(zip(st.out_dims, st.primal_dims))
    )

    # Check sparse index pairing consistency
    matching_sparse_ids = all(
        st.primal_dims[d.other_id - len(st.out_dims)].other_id == d.id
        if isinstance(d, SparseIndex) else True
        for d in st.out_dims
    )

    assert (matching_sizes
            and has_uniqe_dims
            and matching_id
            and matching_sparse_ids
    ), f"{st} is not self-consistent!"

    # TODO: check if val is consistent with tensor structure?


def _get_fully_materialized_shape(st: SparseTensor) -> Sequence[int]:
    """
    Function that returns the shape of a `SparseTensor` object if its `val`
    property would be fully materialized. Indices of size one are inserted 
    for one of the two indices corresponding to a pair of type `SparseIndex`.
    If the `SparseIndex` has val == None, then both are set to one.
    This corresponds to a 

    Args:
        st (SparseTensor): The input tensor we want to materialize
        swap_sparse_dims (bool, optional): Decides which of the pairs of 
            SparseIndices gets the val property. Defaults to False.

    Returns:
        tuple[int]: The fully materialized shape.
    """
    # Compute out_dims full shape-mul
    def out_dim_fn(d: Index) -> int:
        # NOTE we need the case `d.size != st.val.shape[d.val_axis]` because SparseIndices can be matrialized without
        #   the correct d.size property
        if d.val_axis is None or d.size != st.val.shape[d.val_axis]:
            return 1
        else:
            return d.size
        
    out_shape = tuple(out_dim_fn(d) for d in st.out_dims)
           
    # Compute primal_dims full shape
    def primal_dim_fn(d: Index) -> int:
        if isinstance(d, SparseIndex) or d.val_axis is None:
            return 1
        else:
            return d.size

    primal_shape = tuple(primal_dim_fn(d) for d in st.primal_dims)

    return out_shape + primal_shape

    
def _is_pure_dot_product_mul(lhs: SparseTensor, rhs: SparseTensor) -> bool:
    """
    Function that checks if two `SparseTensor` objects are compatible for a
    dot product multiplication. 

    Args:
        lhs (SparseTensor): The left-hand side `SparseTensor` object.
        rhs (SparseTensor): The right-hand side `SparseTensor` object.
    
    Returns:
        bool: Are the tensors compatible for multiplication?
    """
    return all(isinstance(r, DenseIndex) and isinstance(l, DenseIndex)
               for r, l in zip(lhs.primal_dims, rhs.out_dims))


def _is_pure_broadcast_mul(lhs: SparseTensor, rhs: SparseTensor) -> bool:
    """
    Function that checks if two `SparseTensor` objects are compatible for a
    broadcast multiplication.

    Args:
        lhs (SparseTensor): The left-hand side `SparseTensor` object.
        rhs (SparseTensor): The right-hand side `SparseTensor` object.

    Returns:
        bool: Are the tensors compatible for multiplication?
    """
    return all(isinstance(l, SparseIndex) or isinstance(r, SparseIndex)
               for l, r in zip(lhs.primal_dims, rhs.out_dims))

    
def _get_identity_scale(t: SparseTensor):
    return 1.0 if t.val is None else t.val


def _is_identity_tensor(t: SparseTensor) -> bool:
    """True when t is a scaled Kronecker delta with no materialized data array.

    Requires every index to be a SparseIndex with val_axis=None AND
    out_dims[j] paired with primal_dims[j] (in order, same sizes). A permuted
    identity (e.g. a transpose) shares the val_axis=None property but is NOT
    a scalar-scaled identity and must go through the normal multiply path.
    """
    if not (
        all(isinstance(d, SparseIndex) and d.val_axis is None for d in t.dims)
        and (t.val is None or isinstance(t.val, (int, float))
             or (hasattr(t.val, 'size') and t.val.size == 1))
        and not t.pre_transforms
        and not t.post_transforms
        and len(t.out_dims) == len(t.primal_dims)
    ):
        return False
    # Verify each out_dim[j] is directly paired with primal_dim[j] (not permuted).
    return all(
        od.other_id == pd.id and pd.other_id == od.id and od.size == pd.size
        for od, pd in zip(t.out_dims, t.primal_dims)
    )


def _mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    Function that multiplies two `SparseTensor` objects together. The function
    first performs a sequence of checks to guarantee the integrity of both
    `SparseTensor` objects. It then proceeds to check if the two tensors are
    compatible for multiplication. If they are, it performs the right
    multiplicationn type (dot product, broadcast, mixed)
    and returns the resulting `SparseTensor` object.

    Args:
        lhs (SparseTensor): The left-hand side `SparseTensor` object.
        rhs (SparseTensor): The right-hand side `SparseTensor` object.

    Returns:
        SparseTensor: The resulting `SparseTensor` object.
    """
    l = len(lhs.out_dims)
    r = len(rhs.out_dims)
    assert lhs.shape[l:] == rhs.shape[:r], \
        f"{lhs.shape} and {rhs.shape} not compatible for multiplication!"

    if lhs.shape == () and rhs.shape == ():
        # If both tensors are scalars, we can just multiply them directly
        res = SparseTensor((), (), lhs.val * rhs.val)
    elif _is_identity_tensor(lhs):
        if lhs.val is None:
            res = rhs.copy()
        elif rhs.val is None:
            res = rhs.copy(lhs.val)
        else:
            res = rhs.copy(rhs.val * lhs.val)
    elif _is_identity_tensor(rhs):
        if rhs.val is None:
            res = lhs.copy()
        elif lhs.val is None:
            res = lhs.copy(rhs.val)
        else:
            res = lhs.copy(lhs.val * rhs.val)
    else:
        _lhs = lhs.copy()
        _rhs = rhs.copy()
        if _is_pure_dot_product_mul(_lhs, _rhs):
            res = _pure_dot_product_mul(_lhs, _rhs)
        elif _is_pure_broadcast_mul(_lhs, _rhs):
            res = _pure_broadcast_mul(_lhs, _rhs)
        else:
            res = _mixed_mul(_lhs, _rhs)

    _assert_sparse_tensor_consistency(res)
    return res


def _add(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    Function that multiplies two `SparseTensor` objects together. The function
    first performs a sequence of checks to guarantee the integrity of both 
    `SparseTensor` objects. It then proceeds to add both tensors, thereby
    possibly materializing certain sparse indices.

    Args:
        lhs (SparseTensor): The left-hand side `SparseTensor` object.
        rhs (SparseTensor): The right-hand side `SparseTensor` object.

    Returns:
        SparseTensor: The resulting `SparseTensor` object.
    """
    assert lhs.shape == rhs.shape, \
        f"{lhs.shape} and {rhs.shape} not compatible for addition!"

    # Shortcut: when both operands are "scalar * sparse-identity structure"
    # (all val_axis=None, matching dim types/sizes/pairings), their sum is the
    # same structure with val=lhs.val+rhs.val. _sparse_add's general path
    # calls _materialize_axes on scalar vals, which expands to size-1
    # axes and produces a tensor that violates the consistency invariant
    # (val.shape[val_axis] != d.size). This shortcut avoids that path entirely.
    if (lhs.val is not None and rhs.val is not None
            and all(d.val_axis is None for d in lhs.dims)
            and all(d.val_axis is None for d in rhs.dims)
            and len(lhs.out_dims) == len(rhs.out_dims)
            and len(lhs.primal_dims) == len(rhs.primal_dims)
            and all(type(ld) is type(rd) and ld.size == rd.size
                    and (not isinstance(ld, SparseIndex)
                         or ld.other_id == rd.other_id)
                    for ld, rd in zip(lhs.dims, rhs.dims))):
        return lhs.copy(lhs.val + rhs.val)

    res = _sparse_add(lhs, rhs)

    _assert_sparse_tensor_consistency(res)
    return res

def _get_new_val_axis(d: Index, st: SparseTensor) -> int:
    """
    Function that computes the new `val_axis` of a `SparseIndex` object
    so that it's position within the `val` property matches the relative position
    of the corresponding `SparseIndex` object in the `primal_dims` list.
    
    Args:
        d (Index): Index object whose new `val_axis` we want to compute.
        st (SparseTensor): SparseTensor object that contains the `d` object.
        
    Returns:
        int: The new `val_axis` of the `d` object.
    """
    if d.id < d.other_id:
        dims = st.dims[:d.other_id]
    else:
        dims = st.dims[:d.id]

    other_val_axiss = [_d.val_axis for _d in dims if _d.val_axis is not None]
    
    if other_val_axiss:
        return max(other_val_axiss)
    else: 
        return None


def _get_padding(lhs_out_dims: Sequence[Index], 
                 rhs_primal_dims: Sequence[Index]) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Function that calculates how many indices have to be prepended/appended
    to the `val` property of a `SparseTensor` to make it compatible for broadcast
    multiplication with another `SparseTensor`.
    
    Removes excess indices which are artifacts of `SparseTensor` objects.

    Args:
        lhs_out_dims (SparseIndex):
            SparseIndex object whose `val` property we want to multiply with `rhs.val`.
        rhs_primal_dims (SparseIndex):
            SparseIndex object whose `val` property we want to multiply with `lhs.val`.

    Returns:
        Tuple[Sequence[int, ...], Sequence[int, ...]]:
            A tuple of integers that tells us how many indices we have to
            append/prepend to the `val` property of `lhs` and `rhs`.
    """
    # Calculate where we have to add additional indices to rhs.val
    # due to DenseIndices in lhs.out_dims    
    lhs_pad = tuple(1 for d in rhs_primal_dims if isinstance(d, DenseIndex) and d.val_axis is not None)
    rhs_pad = tuple(1 for d in lhs_out_dims if isinstance(d, DenseIndex) and d.val_axis is not None)
    return lhs_pad, rhs_pad


def _assert_broadcast_compatibility(lhs_val: Array, rhs_val: Array):
    """
    Function that checks if two arrays are compatible for broadcast multiplication. 
    
    Args:
        lhs_val (Array): Array that we want to multiply with `rhs_val`.
        rhs_val (Array): Array that we want to multiply with `lhs_val`.
    """
    if lhs_val.ndim == 0 or rhs_val.ndim == 0:
        return  # scalars broadcast with anything
    assert (
        len(lhs_val.shape) == len(rhs_val.shape)
        and all(
            (ls == rs or ls == 1 or rs == 1)
            for (ls, rs) in zip(lhs_val.shape, rhs_val.shape)
        )
    ), f"Shapes {lhs_val.shape} and {rhs_val.shape} not compatible for broadcast multiplication!"


def _get_permutation_from_tensor(st: SparseTensor,
                                 shape: Sequence[int] = None) -> Sequence[int]:
    """
    Function that calculates the permutation of the axes of the `val` property
    so as that `st.val.shape` matches `shape`. This is necessary to enable proper
    broadcasting multiplication.s
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            compute the permutation for.
        shape (Sequence[int], optional): The shape we want to permute the `val`
                                        property of `st` to. Defaults to None.
    
    Returns:
        Sequence[int]: Permutation of the axes of `st.val` so that it matches
                        `shape`.
    """
    shape = shape if shape else st.val.shape
    permutation = [0]*len(st.val.shape)
    
    i = 0
    for d in st.dims:
        if d.val_axis is not None and (isinstance(d, DenseIndex) or d.id < d.other_id):
            permutation[d.val_axis] = i
            i += 1
    return permutation


def _get_val_shape(st: SparseTensor) -> Sequence[int]:
    """
    Function that computes the shape of the `val` property of a `SparseTensor`
    from its corresponding `Index` objects.
    We assume that for `SparseIndices` the corresponding index is at the
    relative position that relates to the entry of the `SparseIndex` in
    the `out_dims` list.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            compute the shape of.
                            
    Returns:
        Sequence[int]: Shape of the `val` property of the `SparseTensor` object.
    """
    shape = [0]*st.val.ndim
    for d in st.out_dims:
        if d.val_axis is not None:
            shape[d.val_axis] = d.size
            
    for d in st.primal_dims:
        if d.val_axis is not None and isinstance(d, DenseIndex):
            shape[d.val_axis] = d.size
    return shape


def _swap_axes(st: SparseTensor) -> SparseTensor:
    """Function that swaps the axes of the `val` property of a `SparseTensor`
    so that the `val_axis`s of SparseIndex objects conincide with the position
    in the `primal_dims` list. This is necessary to enable proper broadcasting 
    multiplication.
    
    Example: 
    The tensor with `out_dims=(SparseIndex(0, 2, 0, 3), DenseIndex(1, 3, 1))`
    and `primal_dims=(DenseIndex(2, 4, 2), SparseIndex(3, 2, 0, 0))` and 
    `val.shape = (2, 3, 4)` have it's `val` array turned into shape (3, 4, 2).
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            swap around for broadcasting multiplication.
    Returns:
        SparseTensor: SparseTensor object with appropriately swapped `val` property.
    """
    transposed_shape = [d.size for d in st.out_dims if isinstance(d, DenseIndex)]
    transposed_shape += [d.size for d in st.primal_dims if d.val_axis is not None]
    
    l = len(st.out_dims)
    for ld in st.out_dims:
        # NOTE: not sure if this is a good solution to the problem here:
        if transposed_shape == _get_val_shape(st):
            break
        if isinstance(ld, SparseIndex) and ld.val_axis is not None:
            new_val_axis = _get_new_val_axis(ld, st)
            for d in st.dims:
                if (d.id != ld.id
                    and d.id != ld.other_id
                    and d.val_axis is not None
                    and ld.val_axis <= d.val_axis <= new_val_axis):
                    d.val_axis -= 1
            ld.val_axis = new_val_axis
            st.primal_dims[ld.other_id-l].val_axis = new_val_axis

    permutation = _get_permutation_from_tensor(st)
    st.val = jnp.transpose(st.val, permutation)
    return st


def _pad_tensors(lhs: SparseTensor, rhs: SparseTensor):
    """
    Function that pads the `val` properties of two `SparseTensor` objects for
    proper broadcast multiplication. It does the following three things:
        1. It appends new axes to the `lhs` tensor for every `DenseIndex` in
            the `rhs.primal_dims` list.
        2. It prepends new axes to the `rhs` tensor for every `DenseIndex` in
            the `lhs.out_dims` list.
        3. It adds new axes to the `lhs.val` for every `SparseIndex` with 
            `val_axis = None` in the `lhs` tensor where `rhs` tensor has a
            `Index` object with `val_axis != None` at the same position in 
            `rhs.out_dims`.
        4. It does the same as in 3. for the `rhs` tensor.
        5. It checks the `val_axis` property of `lhs` and rhs` at the same index.
            If both are None, it does not insert a new axis.
            
    NOTE: The `val_axis` properties of the `Index` objects are changed accordingly.
    
    Example:
    The `lhs` tensor with 
    `out_dims=(SparseIndex(0, 2, 0, 3), DenseIndex(1, 3, 1))`
    `primal_dims=(DenseIndex(2, 4, 2), SparseIndex(3, 2, 0, 0))`
    `val.shape = (2, 3, 4)`
    and the `rhs` tensor with 
    `out_dims=(SparseIndex(0, 4, None, 2), DenseIndex(1, 2, 1))`
    `primal_dims=(SparseIndex(2, 4, None, 0), DenseIndex(3, 3, 0))` 
    `val.shape = (2, 3)` 
    have their `val` properties turned into shapes (3, 4, 2, 1)
    and (1, 1, 2, 3) respectively so that they can be broadcast multiplied.
    
    NOTE: This functions assumes that `_swap_axes` has been applied to the 
    `lhs` tensor before calling this function.
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
    
    Returns:
        tuple[SparseTensor, SparseTensor]: tuple of SparseTensor objects with
                                           appropriately padded `val` properties
                                           and corresponding changes to the
                                           `val_axis` properties.
    """
    lhs_shape, rhs_shape = list(lhs.val.shape), list(rhs.val.shape)
    r = len(rhs.out_dims)
    lhs_pad, rhs_pad = _get_padding(lhs.out_dims, rhs.primal_dims)
    
    ### Update index numbers
    for rd in rhs.dims:
        if rd.val_axis is not None:
            if isinstance(rd, DenseIndex):
                rd.val_axis += len(rhs_pad)
            elif rd.id < rd.other_id:
                    rd.val_axis += len(rhs_pad)
                    primal_dim = rhs.primal_dims[rd.other_id-r]
                    primal_dim.val_axis += len(rhs_pad)
    
    lhs_shape = list(lhs_shape) + list(lhs_pad)
    rhs_shape = list(rhs_pad) + list(rhs_shape)      
          
    ### Add indices where things are sparse    
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if ld.val_axis is None and rd.val_axis is None:
            continue
        # ld is sparse
        if ld.val_axis is None and isinstance(ld, SparseIndex):
            other_val_axis = _get_new_val_axis(ld, lhs)
            if other_val_axis is not None:
                other_val_axis += 1
            else:
                other_val_axis = 0

            lhs_shape.insert(other_val_axis, 1)
            ld.val_axis = other_val_axis
            lhs.out_dims[ld.other_id].val_axis = other_val_axis

            for d in lhs.dims:
                if (d.id != ld.id
                    and d.id != ld.other_id
                    and d.val_axis is not None
                    and d.val_axis >= other_val_axis):
                    d.val_axis += 1
                   
        # rd is sparse
        elif rd.val_axis is None and isinstance(rd, SparseIndex):
            dims = [d.val_axis for d in rhs.out_dims[:rd.id] if d.val_axis is not None]
            new_val_axis = 0
            if dims:
                new_val_axis = max(dims) + 1

            rhs_shape.insert(new_val_axis, 1)
            rd.val_axis = new_val_axis
            rhs.primal_dims[rd.other_id-r].val_axis = new_val_axis

            for d in rhs.dims:
                if (d.id != rd.id
                    and d.id != rd.other_id
                    and d.val_axis is not None
                    and d.val_axis >= new_val_axis):
                    d.val_axis += 1
                        
        # ld is replicating        
        elif ld.val_axis is None and isinstance(ld, DenseIndex):
            new_val_axis = _get_val_axis_when_swapped(lhs, ld.id) # cannot use this here!

            lhs_shape.insert(new_val_axis, 1)
            ld.val_axis = new_val_axis

            for d in lhs.dims:
                if d.id != ld.id:
                    if d.val_axis is not None and d.val_axis >= new_val_axis:
                        d.val_axis += 1
        
        # rd is replicating        
        elif rd.val_axis is None and isinstance(rd, DenseIndex):
            new_val_axis = _get_val_axis(rhs, rd.id)
            # dims = [d.val_axis for d in rhs.out_dims[:rd.id] if d.val_axis is not None]
            # new_val_axis = 0
            # if len(dims) > 0:
            #     new_val_axis = max(dims) + 1
            rhs_shape.insert(new_val_axis, 1)
            rd.val_axis = new_val_axis

            for d in rhs.dims:
                if (d.id != rd.id
                    and d.val_axis is not None
                    and d.val_axis >= new_val_axis):
                    d.val_axis += 1

    # Only do a reshape if the shape differs from the unmodified one
    if lhs_shape != lhs.val.shape:
        lhs.val = lhs.val.reshape(lhs_shape)  
    if rhs_shape != rhs.val.shape: 
        rhs.val = rhs.val.reshape(rhs_shape)   
                    
    return lhs, rhs


def _swap_back_axes(st: SparseTensor) -> SparseTensor:
    """
    After two `SparseTensor` objects have been broadcast multiplied, the
    resulting tensor usually has the `val` not reshaped so that the indices
    of it are sorted in ascending order according to the order in which the
    corresponding indices appear. This function does this.
    
    Example:
    We might end up with a `SparseTensor` object that looks like
    `out_dims=(SparseIndex(0, 2, 1, 3), DenseIndex(1, 3, 2))`
    `primal_dims=(DenseIndex(2, 4, 0), SparseIndex(3, 2, 1, 0))`
    `val.shape = (4, 2, 3)` 
    but we want to have `val.shape = (2, 3, 4)`.
    This function computes the necessary permutation and applies it as a
    `jnp.transpose` to the `val` property.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            swap back around after broadcasting multiplication.
    
    Returns:
        SparseTensor: SparseTensor object with `val` property with indices
                        sorted in ascending order.
    """
    l = len(st.out_dims)
    i = 0
    permutation = [0]*len(st.val.shape)
    for d in st.dims:
        if d.val_axis is not None and (isinstance(d, DenseIndex) or d.id < d.other_id):
            permutation[i] = d.val_axis
            i += 1
     
    st.val = jnp.transpose(st.val, permutation)
    
    i = 0
    for d in st.dims:
        if d.val_axis is not None:
            if isinstance(d, DenseIndex):
                d.val_axis = i
                i += 1
            else:
                if d.id < d.other_id:
                    d.val_axis = i
                    primal_dim = st.primal_dims[d.other_id-l]
                    primal_dim.val_axis = i if primal_dim.val_axis is not None else None
                    i += 1
    return st


def _get_output_tensor(lhs: SparseTensor, 
                        rhs: SparseTensor,
                        val: Array) -> SparseTensor:
    """Function that computes the `out_dims` and `primal_dims` properties
    of a `SparseTensor` object of a broadcast multiplication of two `SparseTensor`
    objects. This is separated from the actual multiplication and broadcasting
    of the `val` properties to make the code more readable. Also in several
    corner cases we actually just need to reassign some `val_axiss` and not
    perform any actual calculations. The approach here also takes care of this
    and saves multiplications by just storing the meta data of some trivial 
    multiplications.
    
    Example:
    TODO put an appropriate example here!
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
        val (Array): The `val` property of the resulting `SparseTensor` object.
    
    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        broadcasting multiplication of `lhs.val` and `rhs.val`.
    """
    new_out_dims, new_primal_dims = [], []
    l, r = len(lhs.out_dims), len(rhs.out_dims)
    
    for ld in lhs.out_dims:
        if isinstance(ld, DenseIndex):
            new_out_dims.append(DenseIndex(ld.id, ld.size, ld.val_axis))
        else:
            # `d` is a SparseIndex and we know it has a corresponding friend
            # in lhs.primal_dims. We now check with what index in rhs.out_dims
            # it will get contracted.
            idx = ld.other_id - l
            rd = rhs.out_dims[idx]
            if isinstance(rd, DenseIndex):
                new_out_dims.append(DenseIndex(ld.id, ld.size, ld.val_axis))
            else:          
                other_id = rd.other_id-r+l
                new_out_dims.append(SparseIndex(ld.id, ld.size, ld.val_axis, other_id))
                new_primal_dims.insert(other_id-l, SparseIndex(other_id, ld.size, ld.val_axis, ld.id))

    # Calculate where we have to add additional indices to lhs.val
    # due to DenseIndices in rhs.primal_dims
    new_dense_dims = []
    for rd in rhs.primal_dims:
        if isinstance(rd, DenseIndex):
            # shift = sum([1 for dim in new_dense_dims if dim <= rd.val_axis])
            new_primal_dims.insert(rd.id-r, DenseIndex(rd.id-r+l, rd.size, rd.val_axis))
        else:
            idx = rd.other_id - r
            ld = lhs.primal_dims[idx]
            if isinstance(ld, DenseIndex):
                new_dense_dims.append(ld.val_axis)
                new_primal_dims.insert(rd.id-r, DenseIndex(rd.id-r+l, ld.size, ld.val_axis))                 
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    

def _pure_broadcast_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor: 
    """Function that executes a pure broadcast multiplication of two `SparseTensor`
    objects. This occurs only if for every pair of `Index` objects  in
    `lhs.primal_dims` and `rhs.out_dims` we have at least one `SparseIndex`.
    In these cases we do not have to perform a full matrix multiplication and 
    get away with simple elementwise multiplication given we broadcast the 
    `val` properties of the `lhs` and `rhs` tensors to the right shape.
    This function takes care of this by swapping axes and padding the `val` 
    properties accordingly.
    
    NOTE: This happens a lot actually!
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            pad for broadcasting multiplication.
    
    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        broadcasting multiplication of `lhs.val` and `rhs.val`.
    """                                          
    ### Calculate output tensor
    if lhs.val is None and rhs.val is None:
        return _get_output_tensor(lhs, rhs, None)
    elif lhs.val is None:
        return _get_output_tensor(lhs, rhs, rhs.val)
    elif rhs.val is None:
        return _get_output_tensor(lhs, rhs, lhs.val)
    else: 
        # Swap left axes if sparse
        lhs = _swap_axes(lhs)      

        # Add padding
        lhs, rhs = _pad_tensors(lhs, rhs)
            
        _assert_broadcast_compatibility(lhs.val, rhs.val)
        new_val = lhs.val * rhs.val
        out = _get_output_tensor(lhs, rhs, new_val)
        res = _swap_back_axes(out)
        return res
    
    
def _get_val_axis(st: SparseTensor, id: int) -> int:
    """Function to get the `val_axis` of both `SparseIndex` objects in the 
    `out_dims` and `primal_dims` lists of a `SparseTensor` object.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val_axis` we want to know.
        id (int): `SparseIndex` object with id `id` we want to compute the 
                `val_axis` for.

    Returns:
        int: The `val_axis` of the `SparseIndex` object with id `id`.
    
    """
    dims = st.dims
    i = 0
    for d in dims[:id]:
        if d.val_axis is not None:
            if isinstance(d, DenseIndex):
                i += 1
            elif d.id < d.other_id:
                    i += 1
    return i


def _get_val_axis_when_swapped(st: SparseTensor, id: int) -> int:
    """ Function to get the `val_axis` of both `SparseIndex` objects in the
    `out_dims` and `primal_dims` lists of a `SparseTensor` object where the axes
    have been swapped.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val_axis` we want to know.
        id (int): `SparseIndex` object with id `id` we want to compute the 
                `val_axis` for.
                
    Returns:
        int: The `val_axis` of the `SparseIndex` object with id `id`.
    """
    dims = st.dims
    i = 0
    for d in dims[:id]:
        if d.val_axis is not None:
            if isinstance(d, DenseIndex):
                i += 1
            elif d.id > d.other_id:
                    i += 1
    return i
    
    
def _replicate_along_axis(st: SparseTensor, ids: Sequence[int]) -> SparseTensor:
    """Function that replicates the `val` property of a `SparseTensor` object
    along a given axis. This is necessary to enable broadcasting multiplication
    of two `SparseTensor` objects where one of them has a `DenseIndex` object
    in its `out_dims` list and the other one has a `SparseIndex` object in
    its `out_dims` list.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            replicate along a given axis.
        axes (Sequence[int]): Axes along which we want to replicate the `val` 
                                property of `st`.
    
    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        replication of `st.val` along `axis`.
    """
    # Expand the indices
    dims = st.dims
    new_dims = []
    for id in ids:
        d = dims[id]

        new_val_axis = _get_val_axis(st, id)
        d.val_axis = new_val_axis
        new_dims.append(new_val_axis)
        for _d in dims[id+1:]:
            if _d.val_axis is not None:
                if _d.val_axis >= new_val_axis:
                    _d.val_axis += 1               
    st.val = jnp.expand_dims(st.val, axis=new_dims)
    
    # Do the tiling
    tiling = []              
    for d in dims:
        if d.val_axis is not None:
            if isinstance(d, DenseIndex):
                if st.val.shape[d.val_axis] != d.size:
                    tiling.append(d.size)
                else:
                    tiling.append(1)
            else:
                if d.id < d.other_id:
                    if st.val.shape[d.val_axis] != d.size:
                        tiling.append(d.size)
                    else:
                        tiling.append(1)

    st.val = jnp.tile(st.val, tiling)

    return st


def _get_contracting_axes(lhs: SparseTensor, rhs: SparseTensor) -> Tuple[Sequence[int], Sequence[int]]:
    """
    Function that computes the axes along which the `val` properties of two
    `SparseTensor` objects will get contracted. This is necessary to enable
    broadcasting multiplication of two `SparseTensor` objects where both of them
    have a `DenseIndex` object in their `out_dims` list.
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            replicate along a given axis.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            replicate along a given axis.
    
    Returns:
        Tuple[Sequence[int], Sequence[int]]: A tuple of sequences of integers that
                                            tell us along which axes the `val`
                                            properties of `lhs` and `rhs` will
                                            get contracted.
    """
    lcontracting_axes, rcontracting_axes = [], []
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if isinstance(ld, DenseIndex) and isinstance(rd, DenseIndex):
            # TODO this causes a bug if both axes are replicating axes
            if ld.val_axis is not None and rd.val_axis is not None:
                lcontracting_axes.append(ld.val_axis)
                rcontracting_axes.append(rd.val_axis)
    return lcontracting_axes, rcontracting_axes


def _pure_dot_product_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    This function takes care of the cases where all indices in 
    `lhs.primal_dims` and `rhs.out_dims` are of type `DenseIndex`.
    Then we only need to do `lax.dot_general` with the right index numbers
    to get the result.

    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            multiply with `rhs.val`.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            multiply with `lhs.val`.

    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        the dense dot-product multiplication of `lhs.val` and `rhs.val`.
    """
    lcontracting_axes, rcontracting_axes = [], []
    lreplication_ids, rreplication_ids = [], []
    new_out_dims = lhs.out_dims
    l = len(lhs.out_dims)
    r = len(rhs.out_dims)
    
    new_primal_dims = []
    i = 0
    for d in rhs.primal_dims:
        if d.val_axis is not None:
            new_primal_dims.append(DenseIndex(d.id-r+l, d.size, l+i))   
            i += 1
        else:
            new_primal_dims.append(DenseIndex(d.id-r+l, d.size, None))

    # Handling contracting variables
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if ld.val_axis is None and rd.val_axis is None:
            lreplication_ids.append(ld.id-l+len(lhs.out_dims))
            rreplication_ids.append(rd.id)
        elif ld.val_axis is None:
            lreplication_ids.append(ld.id-l+len(lhs.out_dims))
        elif rd.val_axis is None:
            rreplication_ids.append(rd.id)
        else:
            lcontracting_axes.append(ld.val_axis)
            rcontracting_axes.append(rd.val_axis)
    # Reshape lhs.val and rhs.val for tiling and replicate along the 
    # respective indices for dpt_product
    if lreplication_ids:
        lhs = _replicate_along_axis(lhs, lreplication_ids)
    if rreplication_ids:
        rhs = _replicate_along_axis(rhs, rreplication_ids)
        
    # Get the contracting axes after the tiling
    if lreplication_ids or rreplication_ids:
        lcontracting_axes, rcontracting_axes = _get_contracting_axes(lhs, rhs)
                    
    # Do the math using dot_general
    index_numbers = ((tuple(lcontracting_axes), tuple(rcontracting_axes)), ((), ()))
    new_val = lax.dot_general(lhs.val, rhs.val, index_numbers)
    
    return SparseTensor(new_out_dims, new_primal_dims, new_val)


def _mixed_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    This is the general case where we have dot-product multiplications as
    well as broadcast multiplications. We first do the dot-product multiplications
    and then the broadcast multiplications by extracting the diagonal of the
    corresponding axes of the resulting tensor of the dot-product contraction.
    
    TODO modularize this!
    TODO write docstring
    TODO write code for DenseIndex with val_axis = None

    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            multiply with `rhs.val`.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            multiply with `lhs.val`.

    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        the mixed multiplication of `lhs.val` and `rhs.val`.
    """
    new_out_dims, new_primal_dims = [], []
    l, r = len(lhs.out_dims), len(rhs.out_dims)
    lcontracting_axes, rcontracting_axes = [], []
    lreplication_ids, rreplication_ids = [], []
    
    # We do contractions first    
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if isinstance(ld, DenseIndex) and isinstance(rd, DenseIndex):
            if ld.val_axis is None and rd.val_axis is None:
                lreplication_ids.append(ld.id-l+len(lhs.out_dims))
                rreplication_ids.append(rd.id)
            elif ld.val_axis is None:
                lreplication_ids.append(ld.id-l+len(lhs.out_dims))
            elif rd.val_axis is None:
                rreplication_ids.append(rd.id)
            else:
                lcontracting_axes.append(ld.val_axis)
                rcontracting_axes.append(rd.val_axis)
    
    if lreplication_ids:
        lhs = _replicate_along_axis(lhs, lreplication_ids)
    if rreplication_ids:
        rhs = _replicate_along_axis(rhs, rreplication_ids)
        
    # Get the contracting axes after the tiling
    if lreplication_ids or rreplication_ids:
        lcontracting_axes, rcontracting_axes = _get_contracting_axes(lhs, rhs)
        
    lbroadcasting_axes, rbroadcasting_axes, pos = [], [], []
    # Then we do broadcasting by extracting diagonals from the contracted tensor
    # TODO: split calculation of Index objects and val property!
    # NOTE: SparseIndex and a DenseIndex with val_axis = None basically get rid
    # of a single jnp.diagonal call!
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if isinstance(ld, SparseIndex) or isinstance(rd, SparseIndex):
            # Here, we have a broadcasting over two tensors that are not just
            # Kronecker deltas
            if ld.val_axis is not None and rd.val_axis is not None \
                and lhs.val.shape[ld.val_axis] == ld.size \
                and rhs.val.shape[rd.val_axis] == rd.size:
                    
                lval_axis = ld.val_axis - sum([1 for lc in lcontracting_axes if lc < ld.val_axis])
                pos.append(lval_axis)
                
                lbroadcasting_axes.append(ld.val_axis)
                rbroadcasting_axes.append(rd.val_axis)
                # The following cases cover ...
                if isinstance(rd, DenseIndex):
                    new_out_dims.insert(ld.other_id, DenseIndex(ld.other_id, ld.size, lval_axis))
                elif isinstance(ld, DenseIndex):
                    new_primal_dims.insert(rd.other_id-r, DenseIndex(rd.other_id-r+l, ld.size, lval_axis))
                else:
                    new_out_dims.insert(ld.other_id, SparseIndex(ld.other_id, ld.size, lval_axis, rd.other_id-r+l))
                    new_primal_dims.insert(rd.other_id-r, SparseIndex(rd.other_id-r+l, ld.size, lval_axis, ld.other_id))
            else:
                # In this case, one of the two tensors we contract is just a 
                # Kronecker delta so we can spare ourselves the contraction
                # and just reflag the index of the new tensor
                # NOTE: here we also cover the cases where we have replicating
                # indices
                # The following cases cover ...
                # TODO simplify this piece of code.
                if isinstance(rd, DenseIndex):
                    # ld sparse
                    val_axis = None
                    if ld.val_axis is not None:
                        val_axis = ld.val_axis
                    elif rd.val_axis is not None:
                        val_axis = rd.val_axis \
                                - sum([1 for rc in rcontracting_axes if rc < rd.val_axis]) \
                                + lhs.val.ndim \
                                - sum([1 for lc in lcontracting_axes]) \
                                - sum([1 for lb in lbroadcasting_axes])
                    new_out_dims.insert(ld.id, DenseIndex(ld.other_id, ld.size, val_axis))
                elif isinstance(ld, DenseIndex):
                    # rd sparse
                    val_axis = None
                    if rd.val_axis is not None:
                        # TODO This thing fails in many cases!
                        val_axis = rd.val_axis \
                                - sum([1 for rc in rcontracting_axes if rc < rd.val_axis]) \
                                + sum([1 for ld in lhs.primal_dims if ld.val_axis]) \
                                - sum([1 for lc in lcontracting_axes])
                    elif ld.val_axis is not None:
                        val_axis = ld.val_axis \
                                - sum([1 for lc in lcontracting_axes if lc < ld.val_axis])
                    new_primal_dims.insert(rd.other_id-r, DenseIndex(rd.other_id-r+l, ld.size, val_axis))
                else:
                    # Sparse-Sparse case where either ld.val_axis is None or rd.val_axis is None
                    val_axis = None
                    if ld.val_axis is not None:
                        val_axis = ld.val_axis
                    elif rd.val_axis is not None:
                        val_axis = rd.val_axis \
                                - sum([1 for rc in rcontracting_axes if rc < rd.val_axis]) \
                                + lhs.val.ndim - sum([1 for lc in lcontracting_axes]) \
                                - sum([1 for lb in lbroadcasting_axes])
                    new_out_dims.insert(ld.other_id, SparseIndex(ld.other_id, ld.size, val_axis, rd.other_id-r+l))
                    new_primal_dims.insert(rd.other_id-r, SparseIndex(rd.other_id-r+l, ld.size, val_axis, ld.other_id))
                    
    if lhs.val is None and rhs.val is None:
        new_val = None        
    elif lhs.val is None:
        new_val = rhs.val
    elif rhs.val is None:
        new_val = lhs.val
    else:      
        dim_numbers = (tuple(lcontracting_axes), tuple(rcontracting_axes))
        batch_indices = (tuple(lbroadcasting_axes), tuple(rbroadcasting_axes)) # we abuse these guys here to handle the SparseIndices
        index_numbers = (dim_numbers, batch_indices)
        new_val = lax.dot_general(lhs.val, rhs.val, index_numbers)

        permutation = [None]*new_val.ndim
        j = 0
        for i in range(new_val.ndim):
            if i < len(pos):
                permutation[pos[i]] = i
            else:
                while permutation[j] is not None:
                    j += 1
                permutation[j] = i
        new_val = jnp.transpose(new_val, permutation)
    
    # Take care of the old indices
    for ld in lhs.out_dims:
        if isinstance(ld, DenseIndex):
            val_axis = None
            if ld.val_axis is not None:
                val_axis = sum([1 for d in new_out_dims[:ld.id] if d.val_axis is not None])
            new_out_dims.insert(ld.id, DenseIndex(ld.id, ld.size, ld.val_axis))

    for rd in rhs.primal_dims:
        if isinstance(rd, DenseIndex):
            val_axis = None
            if rd.val_axis is not None:
                # TODO add documentation here
                num_old_lhs_out_dims = sum([1 for ld in lhs.out_dims 
                                            if isinstance(ld, DenseIndex) \
                                                and ld.val_axis is not None])
                num_old_rhs_out_dims = sum([1 for rd in rhs.out_dims \
                                            if rd.val_axis is not None])
                num_sparse_dims = sum([1 for ld, rd in zip(lhs.primal_dims, rhs.out_dims) 
                                       if (isinstance(ld, SparseIndex) \
                                           or isinstance(rd, SparseIndex)) \
                                           and (ld.val_axis is not None \
                                            or rd.val_axis is not None)])
                val_axis = rd.val_axis + num_old_lhs_out_dims + num_sparse_dims - num_old_rhs_out_dims
            new_primal_dims.insert(rd.id-r, DenseIndex(rd.id-r+l, rd.size, val_axis))
    return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))


def _materialize_axes(st: SparseTensor, dims: Sequence[int]) -> Array:
    """
    Function that materializes the `val` property of a `SparseTensor` object
    along a given set of axes. This is necessary to enable broadcasting multiplication
    of two `SparseTensor` objects where one of them has a `DenseIndex` object
    in its `out_dims` list and the other one has a `SparseIndex` object in
    the corresponding `primal_dims` list or vice versa.
    

    Args:
        st (SparseTensor): The `SparseTensor` object whose `val` property we want
                            to materialize along the axes given in `dims`.
        dims (Sequence[int]): The axes along which we want to materialize the `val`
                                property of `st`.

    Returns:
        Array: The `val` property of `st` materialized along the axes given in `dims`.
    """
    if len(dims) == 0:
        return st.val
    dims = sorted(dims) # reverse=True
    # dims = [d if d <= st.val.ndim else -1 for d in dims]
    _dims, counter = [], st.val.ndim
    for d in dims:
        if d <= st.val.ndim:
            _dims.append(d)
            counter += 1
        else:
            _dims.append(counter)
            counter += 1
    return jnp.expand_dims(st.val, axis=_dims)


def _sparse_add(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    TODO write a function that does the addition of two SparseTensor objects
    and break it down into several functions that do the different steps of the
    process. This is a mess right now!

    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            add to `rhs.val`.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            add to `lhs.val`.

    Returns:
        SparseTensor: SparseTensor object with `val` property resulting from
                        the sparse addition of `lhs.val` and `rhs.val`.
    """    
    assert lhs.shape == rhs.shape, \
        f"Incompatible shapes {lhs.shape} and {rhs.shape} for addition!"
    ldims, rdims = [], []
    new_out_dims, new_primal_dims = [], []
    _lshape, _rshape = [], [] 
    count = 0 
                           
    # Check the indexality of the `out_dims` of both tensors
    for ld, rd in zip(lhs.out_dims, rhs.out_dims):
        if ld.val_axis is None and rd.val_axis is None:
            dim = count
            ldims.append(count)
            rdims.append(count)
            count += 1
        elif ld.val_axis is not None and rd.val_axis is None:
            dim = count
            rdims.append(count)
            count += 1
        elif rd.val_axis is not None and ld.val_axis is None:
            dim = count
            ldims.append(count)
            count += 1
        else:
            dim = count
            count += 1
            
        if isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex) \
            and ld.other_id == rd.other_id:
            new_out_dims.append(SparseIndex(ld.id, ld.size, dim, ld.other_id))
            _lshape.append(1)
            _rshape.append(1)
        elif isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex) \
            and ld.other_id != rd.other_id:
            new_out_dims.append(DenseIndex(ld.id, ld.size, dim))
            _lshape.append(ld.size)
            _rshape.append(ld.size)
        else:
            if isinstance(ld, SparseIndex):
                _rshape.append(1)
                _lshape.append(ld.size)
            elif isinstance(rd, SparseIndex):
                _rshape.append(rd.size)
                _lshape.append(1)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_out_dims.append(DenseIndex(ld.id, ld.size, dim))
            
    # Check the indexality of the `primal_dims` of both tensors        
    for ld, rd in zip(lhs.primal_dims, rhs.primal_dims):                 
        if isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex) \
            and ld.other_id == rd.other_id:
            dim = new_out_dims[ld.other_id].val_axis
            new_primal_dims.append(SparseIndex(ld.id, ld.size, dim, ld.other_id))
            # _lshape.append(1)
            # _rshape.append(1)
        else:            
            if ld.val_axis is None and rd.val_axis is None:
                dim = count
                ldims.append(count)
                rdims.append(count)
                count += 1
            elif ld.val_axis is not None and rd.val_axis is None:
                dim = count
                rdims.append(count)
                count += 1
            elif rd.val_axis is not None and ld.val_axis is None:
                dim = count
                ldims.append(count)
                count += 1
            else:
                dim = count
                count += 1
            
            # TODO something here is not right, there is an apparent asymetry
            # between the cases for ld and rd!
            if isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex) \
                and ld.other_id != rd.other_id:
                _lshape.append(ld.size)
                _rshape.append(ld.size)
            elif isinstance(ld, SparseIndex):
                _lshape.append(ld.size)
                if ld.val_axis is not None:
                    ldims.append(dim)
                _rshape.append(1)
            elif isinstance(rd, SparseIndex):
                _rshape.append(rd.size)
                if rd.val_axis is not None:
                    rdims.append(dim)
                _lshape.append(1)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_primal_dims.append(DenseIndex(ld.id, ld.size, dim))
                        
    lhs_val = _materialize_axes(lhs, ldims)
    rhs_val = _materialize_axes(rhs, rdims)
    
    ltiling = [1]*len(lhs_val.shape)
    rtiling = [1]*len(rhs_val.shape)
    
    _ldims = lhs.dims
    _rdims = rhs.dims
    i = 0
    for ld, rd in zip(_ldims, _rdims):
        if isinstance(ld, DenseIndex) \
            and ld.val_axis is None and rd.val_axis is not None:
            ltiling[i] = ld.size
            i+= 1
        elif isinstance(rd, DenseIndex) \
            and rd.val_axis is None and ld.val_axis is not None:
            rtiling[i] = rd.size
            i += 1
    
    if sum(ltiling) > len(ltiling):
        lhs_val = jnp.tile(lhs_val, ltiling)
    if sum(rtiling) > len(rtiling):
        rhs_val = jnp.tile(rhs_val, rtiling)
        
    # We need to materialize sparse indices for addition
    if sum(_lshape) > len(_lshape):       
        iota = eye_like(_lshape, len(lhs.out_dims))
        lhs_val = iota * lhs_val
    if sum(_rshape) > len(_rshape):
        iota = eye_like(_rshape, len(rhs.out_dims))
        rhs_val = iota*rhs_val
    
    new_val = lhs_val + rhs_val
    return SparseTensor(new_out_dims, new_primal_dims, new_val)
    
    
def get_num_muls(lhs: SparseTensor, rhs: SparseTensor) -> int:
    # Function that counts the number of multiplications done by multiplication
    # of two SparseTensor objects  
    num_muls = 1
    for d in lhs.out_dims:
        if isinstance(d, DenseIndex):
            if d.val_axis is not None:
                num_muls *= d.size
            
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if isinstance(ld, DenseIndex) and isinstance(rd, DenseIndex):
            num_muls *= ld.size
            
        elif isinstance(ld, DenseIndex) and isinstance(rd, SparseIndex):
            num_muls *= ld.size
            
        elif isinstance(ld, SparseIndex) and isinstance(rd, DenseIndex):
            num_muls *= rd.size
            
        elif isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex):
            if ld.val_axis is not None and rd.val_axis is not None:
                m = max([lhs.val.shape[ld.val_axis], rhs.val.shape[rd.val_axis]])
                num_muls *= m
            elif ld.val_axis is not None:
                num_muls *= lhs.val.shape[ld.val_axis]
            elif rd.val_axis is not None:
                num_muls *= rhs.val.shape[rd.val_axis]
            else:
                # Handle multiplications with a multiple of a Kronecker matrix
                if lhs.val is not None and rhs.val is not None:
                    if jnp.size(lhs.val) == 1 and jnp.size(rhs.val) == 1:
                        num_muls *= 1 # ld.size
                    elif jnp.size(lhs.val) == 1:
                        num_muls *= rd.size
                    elif jnp.size(rhs.val) == 1:
                        num_muls *= ld.size

    for d in rhs.primal_dims:
        if isinstance(d, DenseIndex):
            if d.val_axis is not None:
                num_muls *= d.size
                                
    return num_muls
               

# TODO fix this, algorithm might not be correct
def get_num_adds(lhs: SparseTensor, rhs: SparseTensor) -> int:
    """Function that counts the number of multiplications done by addition
    of two `SparseTensor` objects. 
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            add to `rhs.val`.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            add to `lhs.val`.
                            
    Returns:
        int: The number of additions done by addition of `lhs.val` and `rhs.val`.
    """
    num_adds = 1
    
    for ld, rd in zip(lhs.out_dims, rhs.out_dims):
        if isinstance(ld, DenseIndex) and isinstance(rd, DenseIndex):
            num_adds *= ld.size
        elif isinstance(ld, DenseIndex) and isinstance(rd, SparseIndex):
            num_adds *= rd.size
        elif isinstance(ld, SparseIndex) and isinstance(rd, DenseIndex):
            num_adds *= ld.size
        elif isinstance(ld, SparseIndex) and isinstance(rd, SparseIndex):     
            if ld.val_axis is not None and rd.val_axis is not None:    
                num_adds *= ld.size
                
    for ld, rd in zip(lhs.primal_dims, rhs.primal_dims):
        if isinstance(ld, DenseIndex) and isinstance(rd, DenseIndex):
            num_adds *= ld.size
        elif isinstance(ld, DenseIndex) and isinstance(rd, SparseIndex):
            num_adds *= rd.size
        elif isinstance(ld, SparseIndex) and isinstance(rd, DenseIndex):
            num_adds *= ld.size
    return num_adds
    
    