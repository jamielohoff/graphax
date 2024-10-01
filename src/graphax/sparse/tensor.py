"""
Sparse tensor algebra implementation
"""
import copy
from typing import Callable, Sequence, Tuple, Union

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import ShapedArray

from chex import Array

from .utils import eye_like_copy, eye_like


# NOTE a val_dim of None means that we have a possible replication of the tensor 
# along the respective dimension `d.size` times to manage broadcasting 
# operations such as broadcasted additions or multiplications.
# TODO what do we when we have a tensor that consists only of DenseDimensions 
# with val_dim=None?
class DenseDimension:
    id: int
    size: int
    val_dim: int
    
    def __init__(self, id: int, size: int, val_dim: int) -> None:
        self.id = id
        self.size = size
        self.val_dim = val_dim
        
    def __repr__(self) -> str:
        return f"DenseDimension(id={self.id}, size={self.size}, val_dim={self.val_dim})"


# NOTE a val_dim of None means that we have a factored Kronecker delta in
# our tensor at the respective dimensions.
# Also we can have unmatching size and val.shape[d.val_dim] for SparseDimensions
# if the size is 1. This is necessary to enable broadcasting operations.
class SparseDimension:
    id: int
    size: int
    val_dim: int
    other_id: int
    
    def __init__(self, id: int, size: int, val_dim: int, other_id: int) -> None:
        self.id = id
        self.size = size
        self.val_dim = val_dim
        self.other_id = other_id
        
    def __repr__(self) -> str:
        return f"SparseDimension(id={self.id}, size={self.size}, val_dim={self.val_dim}, other_id={self.other_id})"


Dimension = Union[DenseDimension, SparseDimension]
        

class SparseTensor:
    """The `SparseTensor object enables` the representation of sparse tensors
    that 
    if out_dims or primal_dims is empty, this implies a scalar dependent or
    independent variable. 
    if both are empty, then we have a scalar value and everything becomes trivial
    and the `val` field contains the value of the singleton partial
    """
    out_dims: Sequence[Dimension]
    primal_dims: Sequence[Dimension]
    shape: Sequence[int] # True shape of the tensor
    val: ShapedArray
    pre_transforms: Sequence[Callable] 
    post_transforms: Sequence[Callable]
    # NOTE: Document pre_transforms and post_transforms. what about addition?
    # NOTE: We always assume that the dimensions are ordered in ascending order
    
    def __init__(self, 
                out_dims: Sequence[Dimension], 
                primal_dims: Sequence[Dimension], 
                val: ShapedArray, 
                pre_transforms: Sequence[Callable] = [],
                post_transforms: Sequence[Callable] = []) -> None:
                
        self.out_dims = out_dims if type(out_dims) is Tuple else tuple(out_dims)
        self.primal_dims = primal_dims if type(primal_dims) is Tuple else tuple(primal_dims)
        out_shape = [d.size for d in out_dims]
        primal_shape = [d.size for d in primal_dims]
        self.shape = tuple(out_shape + primal_shape)
        self.out_shape = [str(s) for s in out_shape]
        self.primal_shape = [str(s) for s in primal_shape]
        self.val = val
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
            
    def __repr__(self) -> str:
        return f"SparseTensor(\n" \
                f"   shape = (" + f",".join(self.out_shape) + f"|" + f",".join(self.primal_shape) + "),\n" \
                f"   out_dims = {self.out_dims},\n" \
                f"   primal_dims = {self.primal_dims},\n" \
                f"   val = {self.val},\n" \
                f"   pre_transforms = {self.pre_transforms},\n" \
                f"   post_transforms = {self.post_transforms})\n"
                
    def __add__(self, _tensor):
        return _add(self, _tensor)
    
    def __mul__(self, _tensor):
        return _mul(self, _tensor)
                    
    def dense(self, iota: Array) -> Array:
        """
        Materializes tensor to actual dense shape.
        
        TODO add the case where `val_dim = None` for a `DenseDimension` by
        replicating the tensor `d.size` times using `jnp.tile`.

        Args:
            iota (Array): The Kronecker matrix/tensor that is used to 
                materialize the tensor.

        Returns:
            Array: Dense representation of the sparse tensor.
        """
        # Compute shape of the multidimensional eye with which the `val` tensor
        # will get multiplied to manifest the sparse dimensions  
        # If tensor contains SparseDimensions, we have to materialize them
        def eye_dim_fn(d: Dimension) -> int:
            if type(d) is SparseDimension:
                return d.size
            else:
                return 1
            
        eye_shape = [eye_dim_fn(d) for d in self.out_dims+self.primal_dims]
        # If tensor consists only out of Kronecker Delta's, we can just reshape
        # the eye matrix to the shape of the tensor and return it
        if self.val is None: 
            return eye_like_copy(eye_shape, len(self.out_dims), iota)
        
        if self.val.shape == self.shape:
            return self.val
        
        # Catching some corner cases
        if len(self.out_dims) == 0 and len(self.primal_dims) == 0:
            return self.val
        
        shape = _get_fully_materialized_shape(self)   
        
        # Get the tiling for DenseDimensions with val_dim = None, i.e. replicating
        # dimensions
        def tile_dim_fn(d: Dimension) -> int:
            if type(d) is DenseDimension and d.val_dim is None:
                return d.size
            else:
                return 1
        tiling = [tile_dim_fn(d) for d in self.out_dims+self.primal_dims]

        val = self.val.reshape(shape)
        index_map = eye_like_copy(eye_shape, len(self.out_dims), iota)
        return jnp.tile(index_map*val, tiling)
        
    def copy(self, val=None):
        out_dims = copy.deepcopy(self.out_dims)
        primal_dims = copy.deepcopy(self.primal_dims)
        val = self.val if val is None else val
        return SparseTensor(out_dims, primal_dims, val, self.pre_transforms, self.post_transforms)
    
    
def _checkify_tensor(st: SparseTensor) -> bool:
    """Function that validates the consistency of a `SparseTensor` object,
    i.e. checks if the `val` property has the correct shape and if the dimensions
    are ordered correctly and sizes match the shape of `val`.

    Args:
        st (SparseTensor): SparseTensor object we want to validate.

    Returns:
        bool: True if the `SparseTensor` object is consistent.
    """
    # Check if d.size matches val.shape[d.val_dim] for all d
    matching_size = all([d.size == st.val.shape[d.val_dim] 
                        if d.val_dim is not None and type(d) is DenseDimension
                        else True for d in st.out_dims + st.primal_dims])
    
    matching_size += all([d.size == st.val.shape[d.val_dim] or d.size == 1 
                        if d.val_dim is not None and type(d) is SparseDimension
                        else True for d in st.out_dims + st.primal_dims])
        
    unique_out_dims = [d.val_dim for d in st.out_dims if d.val_dim is not None]
    unique_primal_dims = [d.val_dim for d in st.primal_dims if d.val_dim is not None]
    
    is_uniqe_out_dims = len(unique_out_dims) == len(set(unique_out_dims))
    is_uniqe_primal_dims = len(unique_primal_dims) == len(set(unique_primal_dims))
    has_uniqe_dims = is_uniqe_out_dims and is_uniqe_primal_dims
    
    matching_id = True
    matching_sparse_ids = True
    for i, d in enumerate(st.out_dims):
        if i == d.id:
            matching_id *= True
        else:
            matching_id *= False
        if type(d) is SparseDimension:
            _d = st.primal_dims[d.other_id-len(st.out_dims)]
            if d.id == _d.other_id and d.other_id == _d.id:
                matching_sparse_ids *= True
            else:
                matching_sparse_ids *= False
                    
    # TODO speed this up with a list comprehension                
    for i, d in enumerate(st.primal_dims, start=len(st.out_dims)):
        if i == d.id:
            matching_id *= True
        else:
            matching_id *= False
        
    return matching_size and matching_id and has_uniqe_dims and matching_sparse_ids
    

def _get_fully_materialized_shape(st: SparseTensor) -> Tuple[int]:
    """
    Function that returns the shape of a `SparseTensor` object if its 'val' 
    property would be fully materialized. Dimensions of size one are inserted 
    for one of the two dimensions corresponding to a pair of type `SparseDimension'.
    If the `SparseDimension` has val == None, then both are set to one.
    This corresponds to a 

    Args:
        st (SparseTensor): The input tensor we want to materialize
        swap_sparse_dims (bool, optional): Decides which of the pairs of 
            SparseDimensions gets the val property. Defaults to False.

    Returns:
        Tuple[int]: The fully materialized shape.
    """
    # Compute out_dims full shape-mul
    def out_dim_fn(d: Dimension) -> int:
        if d.val_dim is None:
            return 1
        elif d.size != st.val.shape[d.val_dim]:
            return 1 # NOTE we need this case because SparseDimensions can be matrialized without the correct d.size property
        else:
            return d.size
        
    out_shape = [out_dim_fn(d) for d in st.out_dims]
           
    # Compute primal_dims full shape
    def primal_dim_fn(d: Dimension) -> int:
        if type(d) is SparseDimension:
            return 1
        else:
            if d.val_dim is None:
                return 1
            else:
                return d.size
    primal_shape = [primal_dim_fn(d) for d in st.primal_dims]

    return out_shape + primal_shape

    
def _is_pure_dot_product_mul(lhs: SparseTensor, rhs: SparseTensor) -> bool:
    return all([True if type(r) is DenseDimension and type(l) is DenseDimension
                else False for r, l in zip(lhs.primal_dims, rhs.out_dims)])


def _is_pure_broadcast_mul(lhs: SparseTensor, rhs: SparseTensor) -> bool:
    return all([True if type(r) is SparseDimension or type(l) is SparseDimension
                else False for r, l in zip(lhs.primal_dims, rhs.out_dims)])

    
def _mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    TODO docstring
    """                                   
    assert _checkify_tensor(lhs), f"{lhs} is not self-consistent!"
    assert _checkify_tensor(rhs), f"{rhs} is not self-consistent!"
    l = len(lhs.out_dims)
    r = len(rhs.out_dims) 
    assert lhs.shape[l:] == rhs.shape[:r], f"{lhs.shape} and {rhs.shape} "\
                                        "not compatible for multiplication!"

    res = None
    _lhs = lhs.copy()
    _rhs = rhs.copy()
    if lhs.shape == () and rhs.shape == ():
        return SparseTensor((), (), lhs.val*rhs.val)
    elif _is_pure_dot_product_mul(_lhs, _rhs):
        res = _pure_dot_product_mul(_lhs, _rhs)
    elif _is_pure_broadcast_mul(_lhs, _rhs):
        res = _pure_broadcast_mul(_lhs, _rhs)
    else:
        res = _mixed_mul(_lhs, _rhs)

    assert _checkify_tensor(res), f"{res} is not self-consistent!"
    return res


def _add(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    TODO docstring
    """                           
    assert _checkify_tensor(lhs), f"{lhs} is not self-consistent!"
    assert _checkify_tensor(rhs), f"{rhs} is not self-consistent!"

    assert lhs.shape == rhs.shape, f"{lhs.shape} and {rhs.shape} "\
                                        "not compatible for addition!"
    
    res = _sparse_add(lhs, rhs)
    
    assert _checkify_tensor(res), f"{res} is not self-consistent!"
    return res


def _get_other_val_dim(d: Dimension, st: SparseTensor) -> int:
    """Function that computes the new `val_dim` of a `SparseDimension` object
    so that it's position within the `val` property matches the relative position
    of the corresponding `SparseDimension` object in the `primal_dims` list.
    
    Args:
        d (Dimension): Dimension object whose new `val_dim` we want to compute.
        st (SparseTensor): SparseTensor object that contains the `d` object.
        
    Returns:
        int: The new `val_dim` of the `d` object.
    """
    l = len(st.out_dims)
    dims = []
    if d.id < d.other_id:
        dims = st.out_dims + st.primal_dims[:d.other_id-l]
    else:
        dims = st.out_dims + st.primal_dims[:d.id-l]

    other_val_dims = [_d.val_dim for _d in dims if _d.val_dim is not None]
    
    if len(other_val_dims) > 0:
        return max(other_val_dims)
    else: 
        return None


def _get_padding(lhs_out_dims: Sequence[Dimension], 
                rhs_primal_dims: Sequence[Dimension])-> Tuple[int]:
    """Function that calculates how many dimensions have to be prepended/appended
    to the `val` property of a `SparseTensor` to make it compatible for broadcast
    multiplication with another `SparseTensor`.
    
    Removes excess dimensions which are artifacts of `SparseTensor` objects.

    Args:
        lhs (SparseDimension): SparseDimension object whose `val` property we 
                                want to multiply with `rhs.val`.
        rhs (SparseDimension): SparseDimension object whose `val` property we
                                want to multiply with `lhs.val`.

    Returns:
        Tuple[int]: Tuple of integers that tells us how many dimensions we have
                    to append/prepend to the `val` property of `lhs` and `rhs`.
    """
    # Calculate where we have to add additional dimensions to rhs.val
    # due to DenseDimensions in lhs.out_dims    
    lhs_pad = tuple(1 for d in rhs_primal_dims if type(d) is DenseDimension and d.val_dim is not None)
    rhs_pad = tuple(1 for d in lhs_out_dims if type(d) is DenseDimension and d.val_dim is not None)
    return lhs_pad, rhs_pad


def _checkify_broadcast_compatibility(lhs_val: Array, rhs_val: Array) -> bool:
    """Function that checks if two arrays are compatible for broadcast multiplication. 
    
    Args:
        lhs_val (Array): Array that we want to multiply with `rhs_val`.
        rhs_val (Array): Array that we want to multiply with `lhs_val`.
        
    Returns:
        bool: True if the arrays are compatible for broadcast multiplication,
    """
    lhs_shape = lhs_val.shape
    rhs_shape = rhs_val.shape
    assert len(lhs_shape) == len(rhs_shape), f"Shapes {lhs_shape} and {rhs_shape}"\
                                                " not compatible for broadcast_mul!"
    return all([(ls == rs or ls == 1 or rs == 1) 
                for (ls, rs) in zip(lhs_shape, rhs_shape)])
    
    
def _get_permutation_from_tensor(st: SparseTensor,
                                shape: Sequence[int] | None = None) -> Sequence[int]:
    """Function that calculates the permutation of the axes of the `val` property
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
    shape = shape if shape is not None else st.val.shape
    permutation = [0]*len(st.val.shape)
    
    i = 0
    for d in st.out_dims + st.primal_dims:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                permutation[d.val_dim] = i
                i += 1
            else:
                if d.id < d.other_id:
                    permutation[d.val_dim] = i
                    i += 1
    return permutation


def _get_val_shape(st: SparseTensor) -> Sequence[int]:
    """
    Function that computes the shape of the `val` property of a `SparseTensor`
    from its corresponding `Dimension` objects.
    We assume that for `SparseDimensions` the corresponding dimension is at the
    relative position that relates to the entry of the `SparseDimension` in
    the `out_dims` list.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            compute the shape of.
                            
    Returns:
        Sequence[int]: Shape of the `val` property of the `SparseTensor` object.
    """
    shape = [0]*st.val.ndim
    for d in st.out_dims:
        if d.val_dim is not None:
            shape[d.val_dim] = d.size
            
    for d in st.primal_dims:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                shape[d.val_dim] = d.size
    return shape


def _swap_axes(st: SparseTensor) -> SparseTensor:
    """Function that swaps the axes of the `val` property of a `SparseTensor`
    so that the `val_dim`s of SparseDimension objects conincide with the position
    in the `primal_dims` list. This is necessary to enable proper broadcasting 
    multiplication.
    
    Example: 
    The tensor with `out_dims=(SparseDimension(0, 2, 0, 3), DenseDimension(1, 3, 1))`
    and `primal_dims=(DenseDimension(2, 4, 2), SparseDimension(3, 2, 0, 0))` and 
    `val.shape = (2, 3, 4)` have it's `val` array turned into shape (3, 4, 2).
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            swap around for broadcasting multiplication.
    Returns:
        SparseTensor: SparseTensor object with appropriately swapped `val` property.
    """
    transposed_shape = [d.size for d in st.out_dims if type(d) is DenseDimension]
    transposed_shape += [d.size for d in st.primal_dims if d.val_dim is not None]
    
    l = len(st.out_dims)
    for ld in st.out_dims:
        # NOTE: not sure if this is a good solution to the problem at hand:
        if transposed_shape == _get_val_shape(st):
                break
        if type(ld) is SparseDimension and ld.val_dim is not None:
            other_val_dim = _get_other_val_dim(ld, st)
            for d in st.out_dims + st.primal_dims:
                if d.id != ld.id and d.id != ld.other_id:
                    if d.val_dim is not None and d.val_dim >= ld.val_dim and d.val_dim <= other_val_dim:
                        d.val_dim -= 1
            ld.val_dim = other_val_dim
            st.primal_dims[ld.other_id-l].val_dim = other_val_dim

    permutation = _get_permutation_from_tensor(st)
    st.val = jnp.transpose(st.val, permutation)
    return st


def _pad_tensors(lhs: SparseTensor, rhs: SparseTensor):
    """Function that pads the `val` properties of two `SparseTensor` objects for 
    proper broadcast multiplication. It does the following three things:
        1. It appends new axes to the `lhs` tensor for every `DenseDimension` in
            the `rhs.primal_dims` list.
        2. It prepends new axes to the `rhs` tensor for every `DenseDimension` in
            the `lhs.out_dims` list.
        3. It adds new axes to the `lhs.val` for every `SparseDimension` with 
            `val_dim = None` in the `lhs` tensor where `rhs` tensor has a
            `Dimension` object with `val_dim != None` at the same position in 
            `rhs.out_dims`.
        4. It does the same as in 3. for the `rhs` tensor.
        5. It checks the `val_dim` property of `lhs` and rhs` at the same index.
            If both are None, it does not insert a new axis.
            
    NOTE: The `val_dim` properties of the `Dimension` objects are changed accordingly.
    
    Example:
    The `lhs` tensor with 
    `out_dims=(SparseDimension(0, 2, 0, 3), DenseDimension(1, 3, 1))`
    `primal_dims=(DenseDimension(2, 4, 2), SparseDimension(3, 2, 0, 0))`
    `val.shape = (2, 3, 4)`
    and the `rhs` tensor with 
    `out_dims=(SparseDimension(0, 4, None, 2), DenseDimension(1, 2, 1))`
    `primal_dims=(SparseDimension(2, 4, None, 0), DenseDimension(3, 3, 0))` 
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
        Tuple[SparseTensor, SparseTensor]: Tuple of SparseTensor objects with
                                            appropriately padded `val` properties
                                            and corresponding changes to the
                                            `val_dim` properties.
    """
    lhs_shape, rhs_shape = list(lhs.val.shape), list(rhs.val.shape)
    r = len(rhs.out_dims)
    lhs_pad, rhs_pad = _get_padding(lhs.out_dims, rhs.primal_dims)
    
    ### Update dimension numbers
    for rd in rhs.out_dims + rhs.primal_dims:
        if rd.val_dim is not None:
            if type(rd) is DenseDimension:
                rd.val_dim += len(rhs_pad)
            else:
                if rd.id < rd.other_id:
                    rd.val_dim += len(rhs_pad)
                    primal_dim = rhs.primal_dims[rd.other_id-r]
                    primal_dim.val_dim += len(rhs_pad)
    
    lhs_shape = list(lhs_shape) + list(lhs_pad)
    rhs_shape = list(rhs_pad) + list(rhs_shape)      
          
    ### Add dimensions where things are sparse    
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if ld.val_dim is None and rd.val_dim is None:
            continue
        # ld is sparse
        if ld.val_dim is None and type(ld) is SparseDimension:
            other_val_dim = _get_other_val_dim(ld, lhs)
            if other_val_dim is not None:
                other_val_dim += 1
            else:
                other_val_dim = 0

            lhs_shape.insert(other_val_dim, 1)
            ld.val_dim = other_val_dim
            lhs.out_dims[ld.other_id].val_dim = other_val_dim
            for d in lhs.out_dims + lhs.primal_dims:
                if d.id != ld.id and d.id != ld.other_id:
                    if d.val_dim is not None and d.val_dim >= other_val_dim:
                        d.val_dim += 1
                   
        # rd is sparse
        elif rd.val_dim is None and type(rd) is SparseDimension:
            dims = [d.val_dim for d in rhs.out_dims[:rd.id] if d.val_dim is not None]
            new_val_dim = 0
            if len(dims) > 0:
                new_val_dim = max(dims) + 1

            rhs_shape.insert(new_val_dim, 1)
            rd.val_dim = new_val_dim
            rhs.primal_dims[rd.other_id-r].val_dim = new_val_dim

            for d in rhs.out_dims + rhs.primal_dims:
                if d.id != rd.id and d.id != rd.other_id:
                    if d.val_dim is not None and d.val_dim >= new_val_dim:
                        d.val_dim += 1    
                        
        # ld is replicating        
        elif ld.val_dim is None and type(ld) is DenseDimension:
            new_val_dim = _get_val_dim_when_swapped(lhs, ld.id) # cannot use this here!

            lhs_shape.insert(new_val_dim, 1)
            ld.val_dim = new_val_dim

            for d in lhs.out_dims + lhs.primal_dims:
                if d.id != ld.id:
                    if d.val_dim is not None and d.val_dim >= new_val_dim:
                        d.val_dim += 1
        
        # rd is replicating        
        elif rd.val_dim is None and type(rd) is DenseDimension:
            new_val_dim = _get_val_dim(rhs, rd.id)
            # dims = [d.val_dim for d in rhs.out_dims[:rd.id] if d.val_dim is not None]
            # new_val_dim = 0
            # if len(dims) > 0:
            #     new_val_dim = max(dims) + 1
            rhs_shape.insert(new_val_dim, 1)
            rd.val_dim = new_val_dim

            for d in rhs.out_dims + rhs.primal_dims:
                if d.id != rd.id:
                    if d.val_dim is not None and d.val_dim >= new_val_dim:
                        d.val_dim += 1    

    # TODO instead of reshape use `jnp.expand_dims here!`
    ### Needs some serious fixing !#######################################
    # TODO take care of the batched case!
    # Only do a reshape if the shape differs from the unmodified one
    if lhs_shape != lhs.val.shape:
        lhs.val = lhs.val.reshape(lhs_shape)  
    if rhs_shape != rhs.val.shape: 
        rhs.val = rhs.val.reshape(rhs_shape)   
    ######################################################################
                    
    return lhs, rhs


def _swap_back_axes(st: SparseTensor) -> SparseTensor:
    """After two `SparseTensor` objects have been broadcast multiplied, the
    resulting tensor usually has the `val` not reshaped so that the dimensions
    of it are sorted in ascending order according to the order in which the
    corresponding dimensions appear. This function does this.
    
    Example:
    We might end up with a `SparseTensor` object that looks like
    `out_dims=(SparseDimension(0, 2, 1, 3), DenseDimension(1, 3, 2))`
    `primal_dims=(DenseDimension(2, 4, 0), SparseDimension(3, 2, 1, 0))`
    `val.shape = (4, 2, 3)` 
    but we want to have `val.shape = (2, 3, 4)`.
    This function computes the necessary permutation and applies it as a
    `jnp.transpose` to the `val` property.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val` property we want to
                            swap back around after broadcasting multiplication.
    
    Returns:
        SparseTensor: SparseTensor object with `val` property with dimensions
                        sorted in ascending order.
    """
    l = len(st.out_dims)
    i = 0
    permutation = [0]*len(st.val.shape)
    for d in st.out_dims + st.primal_dims:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                permutation[i] = d.val_dim
                i += 1
            else:
                if d.id < d.other_id:
                    permutation[i] = d.val_dim
                    i += 1
                    
    st.val = jnp.transpose(st.val, permutation)
    
    i = 0
    for d in st.out_dims + st.primal_dims:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                d.val_dim = i
                i += 1
            else:
                if d.id < d.other_id:
                    d.val_dim = i
                    primal_dim = st.primal_dims[d.other_id-l]
                    primal_dim.val_dim = i if primal_dim.val_dim is not None else None
                    i += 1
    return st


def _get_output_tensor(lhs: SparseTensor, 
                        rhs: SparseTensor,
                        val: Array | None) -> SparseTensor:
    """Function that computes the `out_dims` and `primal_dims` properties
    of a `SparseTensor` object of a broadcast multiplication of two `SparseTensor`
    objects. This is separated from the actual multiplication and broadcasting
    of the `val` properties to make the code more readable. Also in several
    corner cases we actually just need to reassign some `val_dims` and not
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
        if type(ld) is DenseDimension:
            new_out_dims.append(DenseDimension(ld.id, ld.size, ld.val_dim))
        else:
            # `d` is a SparseDimension and we know it has a corresponding friend
            # in lhs.primal_dims. We now check with what dimension in rhs.out_dims
            # it will get contracted.
            idx = ld.other_id - l
            rd = rhs.out_dims[idx]
            if type(rd) is DenseDimension:
                new_out_dims.append(DenseDimension(ld.id, ld.size, ld.val_dim))
            else:          
                other_id = rd.other_id-r+l
                new_out_dims.append(SparseDimension(ld.id, ld.size, ld.val_dim, other_id))
                new_primal_dims.insert(other_id-l, SparseDimension(other_id, ld.size, ld.val_dim, ld.id))

    # Calculate where we have to add additional dimensions to lhs.val
    # due to DenseDimensions in rhs.primal_dims
    new_dense_dims = []
    for rd in rhs.primal_dims:
        if type(rd) is DenseDimension:
            # shift = sum([1 for dim in new_dense_dims if dim <= rd.val_dim])
            new_primal_dims.insert(rd.id-r, DenseDimension(rd.id-r+l, rd.size, rd.val_dim))
        else:
            idx = rd.other_id - r
            ld = lhs.primal_dims[idx]
            if type(ld) is DenseDimension:
                new_dense_dims.append(ld.val_dim)
                new_primal_dims.insert(rd.id-r, DenseDimension(rd.id-r+l, ld.size, ld.val_dim))                 
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    

def _pure_broadcast_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor: 
    """Function that executes a pure broadcast multiplication of two `SparseTensor`
    objects. This occurs only if for every pair of `Dimension` objects  in
    `lhs.primal_dims` and `rhs.out_dims` we have at least one `SparseDimension`.
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
            
        assert _checkify_broadcast_compatibility(lhs.val, rhs.val), f"Shapes {lhs.val.shape} and {rhs.val.shape} not compatible for broadcast multiplication!"
        new_val = lhs.val * rhs.val
        out = _get_output_tensor(lhs, rhs, new_val)
        res = _swap_back_axes(out)
        return res
    
    
def _get_val_dim(st: SparseTensor, id: int) -> int:
    """Function to get the `val_dim` of both `SparseDimension` objects in the 
    `out_dims` and `primal_dims` lists of a `SparseTensor` object.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val_dim` we want to know.
        id (int): `SparseDimension` object with id `id` we want to compute the 
                `val_dim` for.

    Returns:
        int: The `val_dim` of the `SparseDimension` object with id `id`.
    
    """
    dims = st.out_dims + st.primal_dims
    i = 0
    for d in dims[:id]:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                i += 1
            else:
                if d.id < d.other_id:
                    i += 1
    return i


def _get_val_dim_when_swapped(st: SparseTensor, id: int) -> int:
    """ Function to get the `val_dim` of both `SparseDimension` objects in the
    `out_dims` and `primal_dims` lists of a `SparseTensor` object where the axes
    have been swapped.
    
    Args:
        st (SparseTensor): SparseTensor object whose `val_dim` we want to know.
        id (int): `SparseDimension` object with id `id` we want to compute the 
                `val_dim` for.
                
    Returns:
        int: The `val_dim` of the `SparseDimension` object with id `id`.
    """
    dims = st.out_dims + st.primal_dims
    i = 0
    for d in dims[:id]:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                i += 1
            else:
                if d.id > d.other_id:
                    i += 1
    return i
    
    
def _replicate_along_axis(st: SparseTensor, ids: Sequence[int]) -> SparseTensor:
    """Function that replicates the `val` property of a `SparseTensor` object
    along a given axis. This is necessary to enable broadcasting multiplication
    of two `SparseTensor` objects where one of them has a `DenseDimension` object
    in its `out_dims` list and the other one has a `SparseDimension` object in
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
    # Expand the dimensions
    dims = st.out_dims + st.primal_dims
    new_dims = []
    for id in ids:
        d = dims[id]

        new_val_dim = _get_val_dim(st, id)
        d.val_dim = new_val_dim
        new_dims.append(new_val_dim)
        for _d in dims[id+1:]:
            if _d.val_dim is not None:
                if _d.val_dim >= new_val_dim:
                    _d.val_dim += 1               
    st.val = jnp.expand_dims(st.val, axis=new_dims)
    
    # Do the tiling
    tiling = []              
    for d in dims:
        if d.val_dim is not None:
            if type(d) is DenseDimension:
                if st.val.shape[d.val_dim] != d.size:
                    tiling.append(d.size)
                else:
                    tiling.append(1)
            else:
                if d.id < d.other_id:
                    if st.val.shape[d.val_dim] != d.size:
                        tiling.append(d.size)
                    else:
                        tiling.append(1)

    st.val = jnp.tile(st.val, tiling)

    return st


def _get_contracting_axes(lhs: SparseTensor, rhs: SparseTensor) -> Tuple[Sequence[int], Sequence[int]]:
    """Function that computes the axes along which the `val` properties of two
    `SparseTensor` objects will get contracted. This is necessary to enable
    broadcasting multiplication of two `SparseTensor` objects where both of them
    have a `DenseDimension` object in their `out_dims` list.
    
    Args:
        lhs (SparseTensor): SparseTensor object whose `val` property we want to
                            replicate along a given axis.
        rhs (SparseTensor): SparseTensor object whose `val` property we want to
                            replicate along a given axis.
    
    Returns:
        Tuple[Sequence[int], Sequence[int]]: Tuple of sequences of integers that
                                            tell us along which axes the `val`
                                            properties of `lhs` and `rhs` will
                                            get contracted.
    """
    lcontracting_axes, rcontracting_axes = [], []
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            if ld.val_dim is not None and rd.val_dim is not None:
                lcontracting_axes.append(ld.val_dim)
                rcontracting_axes.append(rd.val_dim)
    return lcontracting_axes, rcontracting_axes


def _pure_dot_product_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """This function takes care of the cases where all dimensions in 
    `lhs.primal_dims` and `rhs.out_dims` are of type `DenseDimension`.
    Then we only need to do `lax.dot_general` wiht the right dimension numbers
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
    print("dense mul before", lhs.val.shape, rhs.val.shape)
    lcontracting_axes, rcontracting_axes = [], []
    lreplication_ids, rreplication_ids = [], []
    new_out_dims = lhs.out_dims
    l = len(lhs.out_dims)
    r = len(rhs.out_dims)
    def _is_none(idx, d):
        return idx if d.val_dim is not None else None
    
    new_primal_dims = []
    i = 0
    for d in rhs.primal_dims:
        if d.val_dim is not None:
            new_primal_dims.append(DenseDimension(d.id-r+l, d.size, l+i))   
            i += 1
        else:
            new_primal_dims.append(DenseDimension(d.id-r+l, d.size, None))

    # Handling contracting variables
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if ld.val_dim is None and rd.val_dim is None:
            lreplication_ids.append(ld.id-l+len(lhs.out_dims))
            rreplication_ids.append(rd.id)
        elif ld.val_dim is None:
            lreplication_ids.append(ld.id-l+len(lhs.out_dims))
        elif rd.val_dim is None:
            rreplication_ids.append(rd.id)
        else:
            lcontracting_axes.append(ld.val_dim)
            rcontracting_axes.append(rd.val_dim)
    # Reshape lhs.val and rhs.val for tiling and replicate along the 
    # respective dimensions for dpt_product
    if len(lreplication_ids) > 0:
        lhs = _replicate_along_axis(lhs, lreplication_ids)
    if len(rreplication_ids) > 0:
        rhs = _replicate_along_axis(rhs, rreplication_ids)
        
    # Get the contracting axes after the tiling
    if len(lreplication_ids) > 0 or len(rreplication_ids) > 0:
        lcontracting_axes, rcontracting_axes = _get_contracting_axes(lhs, rhs)
                    
    # Do the math using dot_general
    dimension_numbers = (tuple(lcontracting_axes), tuple(rcontracting_axes))
    dimension_numbers = (dimension_numbers, ((), ()))
    print("dense mul after", lhs.val.shape, rhs.val.shape, dimension_numbers)
    new_val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
    
    return SparseTensor(new_out_dims, new_primal_dims, new_val)


def _mixed_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """This is the general case where we have dot-product multiplications as
    well as broadcast multiplications. We first do the dot-product multiplications
    and then the broadcast multiplications by extracting the diagonal of the
    corresponding axes of the resulting tensor of the dot-product contraction.
    
    TODO modularize this!
    TODO write docstring
    TODO write code for DenseDimension with val_dim = None

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
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            if ld.val_dim is None and rd.val_dim is None:
                lreplication_ids.append(ld.id-l+len(lhs.out_dims))
                rreplication_ids.append(rd.id)
            elif ld.val_dim is None:
                lreplication_ids.append(ld.id-l+len(lhs.out_dims))
            elif rd.val_dim is None:
                rreplication_ids.append(rd.id)
            else:
                lcontracting_axes.append(ld.val_dim)
                rcontracting_axes.append(rd.val_dim)
    
    if len(lreplication_ids) > 0:
        lhs = _replicate_along_axis(lhs, lreplication_ids)
    if len(rreplication_ids) > 0:
        rhs = _replicate_along_axis(rhs, rreplication_ids)
        
    # Get the contracting axes after the tiling
    if len(lreplication_ids) > 0 or len(rreplication_ids) > 0:
        lcontracting_axes, rcontracting_axes = _get_contracting_axes(lhs, rhs)
        
    lbroadcasting_axes, rbroadcasting_axes, pos = [], [], []
    # Then we do broadcasting by extracting diagonals from the contracted tensor
    # TODO: split calculation of Dimension objects and val property!
    # NOTE: SparseDimension and a DenseDimension with val_dim = None basically get rid
    # of a single jnp.diagonal call!
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is SparseDimension or type(rd) is SparseDimension:
            # Here, we have a broadcasting over two tensors that are not just
            # Kronecker deltas
            if ld.val_dim is not None and rd.val_dim is not None and lhs.val.shape[ld.val_dim] == ld.size and rhs.val.shape[rd.val_dim] == rd.size:
                lval_dim = ld.val_dim - sum([1 for lc in lcontracting_axes if lc < ld.val_dim])
                pos.append(lval_dim)
                
                lbroadcasting_axes.append(ld.val_dim)
                rbroadcasting_axes.append(rd.val_dim)
                # The following cases cover ...
                if type(rd) is DenseDimension:
                    new_out_dims.insert(ld.other_id, DenseDimension(ld.other_id, ld.size, lval_dim))
                elif type(ld) is DenseDimension:
                    new_primal_dims.insert(rd.other_id-r, DenseDimension(rd.other_id-r+l, ld.size, lval_dim))
                else:
                    new_out_dims.insert(ld.other_id, SparseDimension(ld.other_id, ld.size, lval_dim, rd.other_id-r+l))
                    new_primal_dims.insert(rd.other_id-r, SparseDimension(rd.other_id-r+l, ld.size, lval_dim, ld.other_id))
            else:
                # In this case, one of the two tensors we contract is just a 
                # Kronecker delta so we can spare ourselves the contraction
                # and just reflag the dimension of the new tensor
                # NOTE: here we also cover the cases where we have replicating
                # dimensions
                # The following cases cover ...
                # TODO simplify this piece of code.
                if type(rd) is DenseDimension:
                    # ld sparse
                    val_dim = None
                    if ld.val_dim is not None:
                        val_dim = ld.val_dim
                    else:
                        val_dim = rd.val_dim \
                                - sum([1 for rc in rcontracting_axes if rc < rd.val_dim]) \
                                + lhs.val.ndim - sum([1 for lc in lcontracting_axes]) \
                                - sum([1 for lb in lbroadcasting_axes])
                    new_out_dims.insert(ld.id, DenseDimension(ld.other_id, ld.size, val_dim))
                elif type(ld) is DenseDimension:
                    # rd sparse
                    val_dim = None
                    if rd.val_dim is not None:
                        val_dim = rd.val_dim - sum([1 for rc in rcontracting_axes if rc < rd.val_dim]) + lhs.val.ndim - sum([1 for lc in lcontracting_axes])
                    else:
                        val_dim = ld.val_dim - sum([1 for lc in lcontracting_axes if lc < ld.val_dim])
                    new_primal_dims.insert(rd.other_id-r, DenseDimension(rd.other_id-r+l, ld.size, val_dim))
                else:
                    val_dim = None
                    if ld.val_dim is not None:
                        val_dim = ld.val_dim
                    elif rd.val_dim is not None:
                        val_dim = rd.val_dim \
                                - sum([1 for rc in rcontracting_axes if rc < rd.val_dim]) \
                                + lhs.val.ndim - sum([1 for lc in lcontracting_axes]) \
                                - sum([1 for lb in lbroadcasting_axes])
                    new_out_dims.insert(ld.other_id, SparseDimension(ld.other_id, ld.size, val_dim, rd.other_id-r+l))
                    new_primal_dims.insert(rd.other_id-r, SparseDimension(rd.other_id-r+l, ld.size, val_dim, ld.other_id))
                    
    dimension_numbers = (tuple(lcontracting_axes), tuple(rcontracting_axes))
    batch_dimensions = (tuple(lbroadcasting_axes), tuple(rbroadcasting_axes)) # we abuse these guys here to handle the SparseDimensions
    dimension_numbers = (dimension_numbers, batch_dimensions)
    new_val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
    
    permutation =[None]*new_val.ndim
    j = 0
    for i in range(new_val.ndim):
        if i < len(pos):
            permutation[pos[i]] = i
        else:
            while permutation[j] is not None:
                j += 1
            permutation[j] = i
    new_val = jnp.transpose(new_val, permutation)
    
    # Take care of the old dimensions
    for ld in lhs.out_dims:
        if type(ld) is DenseDimension:
            val_dim = None
            if ld.val_dim is not None:
                val_dim = sum([1 for d in new_out_dims[:ld.id] if d.val_dim is not None])
            new_out_dims.insert(ld.id, DenseDimension(ld.id, ld.size, ld.val_dim))
    
    for rd in rhs.primal_dims:
        if type(rd) is DenseDimension:
            val_dim = None
            if rd.val_dim is not None:
                # TODO add documentation here
                num_old_lhs_out_dims = sum([1 for ld in lhs.out_dims 
                                            if type(ld) is DenseDimension and ld.val_dim is not None])
                num_old_rhs_out_dims = sum([1 for rd in rhs.out_dims if rd.val_dim is not None])
                num_sparse_dims = sum([1 for ld, rd in zip(lhs.primal_dims, rhs.out_dims) 
                                       if (type(ld) is SparseDimension or type(rd) is SparseDimension) 
                                       and (ld.val_dim is not None or rd.val_dim is not None)])
                val_dim = rd.val_dim + num_old_lhs_out_dims + num_sparse_dims - num_old_rhs_out_dims
            new_primal_dims.insert(rd.id-r, DenseDimension(rd.id-r+l, rd.size, val_dim))

    return _swap_back_axes(SparseTensor(new_out_dims, new_primal_dims, new_val))


def _materialize_dimensions(st: SparseTensor, dims: Sequence[int]) -> Array:
    """Function that materializes the `val` property of a `SparseTensor` object
    along a given set of axes. This is necessary to enable broadcasting multiplication
    of two `SparseTensor` objects where one of them has a `DenseDimension` object
    in its `out_dims` list and the other one has a `SparseDimension` object in
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
    """TODO write a function that does the addition of two SparseTensor objects
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
    assert lhs.shape == rhs.shape, f"Incompatible shapes {lhs.shape} and {rhs.shape} for addition!"
    ldims, rdims = [], []
    new_out_dims, new_primal_dims = [], []
    _lshape, _rshape = [], [] 
    count = 0 
                           
    # Check the dimensionality of the 'out_dims' of both tensors
    for ld, rd in zip(lhs.out_dims, rhs.out_dims):
        if ld.val_dim is None and rd.val_dim is None:
            dim = count
            ldims.append(count)
            rdims.append(count)
            count += 1
        elif ld.val_dim is not None and rd.val_dim is None:
            dim = count
            rdims.append(count)
            count += 1
        elif rd.val_dim is not None and ld.val_dim is None:
            dim = count
            ldims.append(count)
            count += 1
        else:
            dim = count
            count += 1
            
        if type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id == rd.other_id:
            new_out_dims.append(SparseDimension(ld.id, ld.size, dim, ld.other_id))
            _lshape.append(1)
            _rshape.append(1)
        elif type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id != rd.other_id:
            new_out_dims.append(DenseDimension(ld.id, ld.size, dim))
            _lshape.append(ld.size)
            _rshape.append(ld.size)
        else:
            if type(ld) is SparseDimension:
                _rshape.append(1)
                _lshape.append(ld.size)
            elif type(rd) is SparseDimension:
                _rshape.append(rd.size)
                _lshape.append(1)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_out_dims.append(DenseDimension(ld.id, ld.size, dim))
            
    # Check the dimensionality of the 'primal_dims' of both tensors        
    for (ld, rd) in zip(lhs.primal_dims, rhs.primal_dims):                 
        if type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id == rd.other_id:
            dim = new_out_dims[ld.other_id].val_dim
            new_primal_dims.append(SparseDimension(ld.id, ld.size, dim, ld.other_id))
            # _lshape.append(1)
            # _rshape.append(1)
        else:            
            if ld.val_dim is None and rd.val_dim is None:
                dim = count
                ldims.append(count)
                rdims.append(count)
                count += 1
            elif ld.val_dim is not None and rd.val_dim is None:
                dim = count
                rdims.append(count)
                count += 1
            elif rd.val_dim is not None and ld.val_dim is None:
                dim = count
                ldims.append(count)
                count += 1
            else:
                dim = count
                count += 1
            
            # TODO something here is not right, there is an apparent asymetry
            # between the cases for ld and rd!
            if type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id != rd.other_id:
                _lshape.append(ld.size)
                _rshape.append(ld.size)
            elif type(ld) is SparseDimension:
                _lshape.append(ld.size)
                if ld.val_dim is not None:
                    ldims.append(dim) # -1 # TODO fix this error!
                _rshape.append(1)
            elif type(rd) is SparseDimension:
                _rshape.append(rd.size)
                if rd.val_dim is not None:
                    rdims.append(dim) # -1 # TODO fix this error!
                _lshape.append(1)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_primal_dims.append(DenseDimension(ld.id, ld.size, dim))
                        
    lhs_val = _materialize_dimensions(lhs, ldims)
    rhs_val = _materialize_dimensions(rhs, rdims)
    
    ltiling = [1]*len(lhs_val.shape)
    rtiling = [1]*len(rhs_val.shape)
    
    _ldims = lhs.out_dims + lhs.primal_dims
    _rdims = rhs.out_dims + rhs.primal_dims
    i = 0
    for (ld, rd) in zip(_ldims, _rdims):
        if type(ld) is DenseDimension and ld.val_dim is None and rd.val_dim is not None:
            ltiling[i] = ld.size
            i+= 1
        elif type(rd) is DenseDimension and rd.val_dim is None and ld.val_dim is not None:
            rtiling[i] = ld.size
            i += 1
    
    if sum(ltiling) > len(ltiling):
        lhs_val = jnp.tile(lhs_val, ltiling)
    if sum(rtiling) > len(rtiling):
        rhs_val = jnp.tile(rhs_val, rtiling)
        
    # We need to materialize sparse dimensions for addition
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
        if type(d) is DenseDimension:
            if d.val_dim is not None:
                num_muls *= d.size
            
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            num_muls *= ld.size
            
        elif type(ld) is DenseDimension and type(rd) is SparseDimension:
            num_muls *= ld.size
            
        elif type(ld) is SparseDimension and type(rd) is DenseDimension:
            num_muls *= rd.size
            
        elif type(ld) is SparseDimension and type(rd) is SparseDimension:
            if ld.val_dim is not None and rd.val_dim is not None:
                m = max([lhs.val.shape[ld.val_dim], rhs.val.shape[rd.val_dim]])
                num_muls *= m
            elif ld.val_dim is not None:
                num_muls *= lhs.val.shape[ld.val_dim]
            elif rd.val_dim is not None:
                num_muls *= rhs.val.shape[rd.val_dim]
            else:
                # Handle multiplications with a multiple of a Kronecker matrix
                if lhs.val is not None and rhs.val is not None:
                    if lhs.val.size == 1 and rhs.val.size == 1:
                        num_muls *= 1 # ld.size
                    elif lhs.val.size == 1:
                        num_muls *= rd.size
                    elif rhs.val.size == 1:
                        num_muls *= ld.size

    for d in rhs.primal_dims:
        if type(d) is DenseDimension:
            if d.val_dim is not None:
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
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            num_adds *= ld.size
        elif type(ld) is DenseDimension and type(rd) is SparseDimension:
            num_adds *= rd.size
        elif type(ld) is SparseDimension and type(rd) is DenseDimension:
            num_adds *= ld.size
        elif type(ld) is SparseDimension and type(rd) is SparseDimension:     
            if ld.val_dim is not None and rd.val_dim is not None:    
                num_adds *= ld.size
                
    for ld, rd in zip(lhs.primal_dims, rhs.primal_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            num_adds *= ld.size
        elif type(ld) is DenseDimension and type(rd) is SparseDimension:
            num_adds *= rd.size
        elif type(ld) is SparseDimension and type(rd) is DenseDimension:
            num_adds *= ld.size
    return num_adds
    
    