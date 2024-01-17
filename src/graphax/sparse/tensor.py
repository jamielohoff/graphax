"""
Sparse tensor algebra implementation
"""
import copy
from typing import Callable, Optional, Sequence, Tuple, Union, NamedTuple

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import ShapedArray

from chex import Array

from .utils import eye_like_copy, eye_like


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
    """
    TODO docstring
    if out_dims or primal_dims is empty, this implies a scalar dependent or
    independent variable. 
    if both are empty, then we have a scalar value and everything becomes trivial
    and the `val` field contains the value of the singleton partial
    """
    out_dims: Sequence[Dimension]
    primal_dims: Sequence[Dimension]
    shape: Sequence[int] # True shape of the tensor
    val: ShapedArray
    copy_gradient_fn: Callable
    # NOTE: We always assume that the dimensions are ordered in ascending order
    
    def __init__(self, 
                out_dims: Sequence[Dimension], 
                primal_dims: Sequence[Dimension], 
                val: ShapedArray, 
                copy_gradient_fn: Optional[Callable] = None) -> None:
                
        self.out_dims = out_dims if type(out_dims) is Tuple else tuple(out_dims)
        self.primal_dims = primal_dims if type(primal_dims) is Tuple else tuple(primal_dims)
        out_shape = [d.size for d in out_dims]
        primal_shape = [d.size for d in primal_dims]
        self.shape = tuple(out_shape + primal_shape)
        self.val = val
        self.copy_gradient_fn = copy_gradient_fn
            
    def __repr__(self) -> str:
        return f"SparseTensor: \n" \
                f"   shape = {self.shape}\n" \
                f"   out_dims = {self.out_dims}\n" \
                f"   primal_dims = {self.primal_dims}\n" \
                f"   val = {self.val}\n" \
                f"   copy_gradient_fn = {self.copy_gradient_fn}\n"
                
    def __add__(self, _tensor):
        return _add(self, _tensor)
    
    def __mul__(self, _tensor):
        return _mul(self, _tensor)
        
    def materialize_dimensions(self, dims: Sequence[int]) -> Array:
        if len(dims) == 0:
            return self.val
        dims = sorted(dims, reverse=True)
        dims = [d if d <= self.val.ndim else -1 for d in dims]
        return jnp.expand_dims(self.val, axis=dims)
        
    def full(self, iota: Array) -> Array:
        """
        Materializes tensor to actual shape where Dimensions with val_dim=None
        are materialized as 1

        Args:
            iota (Array): _description_

        Returns:
            Array: _description_
        """
        # Compute shape of the multidimensional eye with which the `val` tensor
        # will get multiplied to manifest the sparse dimensions  
        def eye_dim_fn(d: Dimension) -> int:
            if type(d) is SparseDimension:
                return d.size
            else:
                return 1
        eye_shape = [eye_dim_fn(d) for d in self.out_dims+self.primal_dims]
        
        if self.val is None: 
            return eye_like_copy(eye_shape, len(self.out_dims), iota)
            
        if self.val.shape == self.shape:
            return self.val
        
        if len(self.out_dims) == 0 or len(self.primal_dims) == 0:
            return self.val
        
        shape = _get_fully_materialized_shape(self)   

        val = self.val.reshape(shape)
        index_map = jnp.bool_(eye_like_copy(eye_shape, len(self.out_dims), iota))
        return jnp.where(index_map, val, 0.)
    
    
def _checkify_tensor(st: SparseTensor) -> bool:
    """Function that validates the consistency of a `SparseTensor` object,
    i.e. checks if the `val` property has the correct shape and if the dimensions
    are ordered correctly and sizes match the shape of `val`.

    Args:
        st (SparseTensor): _description_

    Returns:
        bool: _description_
    """
    return all([d.size == st.val.shape[d.val_dim] if d.val_dim is not None 
                else True for d in st.out_dims + st.primal_dims])
    

def _get_fully_materialized_shape(tensor: SparseTensor) -> Tuple[int]:
    """
    Function that returns the shape of a `SparseTensor` object if its 'val' 
    property would be fully materialized. Dimensions of size one are inserted 
    for one of the two dimensions corresponding to a pair of type `SparseDimension'.
    If the `SparseDimension` has val == None, then both are set to one.
    This corresponds to a 

    Args:
        tensor (SparseTensor): The input tensor we want to materialize
        swap_sparse_dims (bool, optional): Decides which of the pairs of 
            SparseDimensions gets the val property. Defaults to False.

    Returns:
        Tuple[int]: The fully materialized shape.
    """
    # Compute out_dims full shape
    def out_dim_fn(d: Dimension) -> int:
        if type(d) is SparseDimension:
            if d.val_dim is None:
                return 1
            else:
                return d.size
        else:
            return d.size
        
    out_shape = [out_dim_fn(d) for d in tensor.out_dims]
           
    # Compute primal_dims full shape
    def primal_dim_fn(d: Dimension) -> int:
        if type(d) is SparseDimension:
            return 1
        else:
            return d.size
    primal_shape = [primal_dim_fn(d) for d in tensor.primal_dims]

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
    _lhs = copy.deepcopy(lhs) # TODO replace this deepcopy hack
    _rhs = copy.deepcopy(rhs)
    if _is_pure_dot_product_mul(_lhs, _rhs):
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
        lhs (SparseDimension): _description_
        rhs (SparseDimension): _description_

    Returns:
        Tuple[int]: _description_
    """
    # Calculate where we have to add additional dimensions to rhs.val
    # due to DenseDimensions in lhs.out_dims    
    lhs_pad = tuple(1 for d in rhs_primal_dims if type(d) is DenseDimension)
    rhs_pad = tuple(1 for d in lhs_out_dims if type(d) is DenseDimension)
    return lhs_pad, rhs_pad


def _checkify_broadcast_compatibility(lhs_val: Array, rhs_val: Array) -> bool:
    """
    Function that checks if two arrays are compatible for broadcast multiplication. 
    """
    lhs_shape = lhs_val.shape
    rhs_shape = rhs_val.shape
    assert len(lhs_shape) == len(rhs_shape), f"Shapes {lhs_shape} and {rhs_shape}"\
                                                "not compatible for broadcast_mul!"
    return all([(ls == rs or ls == 1 or rs == 1) 
                for (ls, rs) in zip(lhs_shape, rhs_shape)])
    
    
def _get_permutation_from_tensor(st: SparseTensor,
                                shape: Sequence[int] | None = None) -> Sequence[int]:
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
    shape = [0]*st.val.ndim
    for d in st.out_dims:
        if d.val_dim is not None:
            shape[d.val_dim] = d.size
            
    for d in st.primal_dims:
        if type(d) is DenseDimension:
            shape[d.val_dim] = d.size
    return shape


def _swap_axes(st: SparseTensor) -> SparseTensor:
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
    # TODO extensive documentation
    lhs_shape, rhs_shape = list(lhs.val.shape), list(rhs.val.shape)
    l, r = len(lhs.out_dims), len(rhs.out_dims)
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
        if ld.val_dim is None:
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
        elif rd.val_dim is None:
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
                    primal_dim.val_dim = i
                    i += 1
    return st


def _get_output_tensor(lhs: SparseTensor, 
                        rhs: SparseTensor,
                        val: Array | None) -> SparseTensor:
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
    for rd in rhs.primal_dims:
        if type(rd) is DenseDimension:
            new_primal_dims.insert(rd.id-r, DenseDimension(rd.id, rd.size, rd.val_dim))
        else:
            idx = rd.other_id
            ld = lhs.primal_dims[idx]
            if type(ld) is DenseDimension:
                new_primal_dims.insert(rd.id-r, DenseDimension(rd.id, ld.size, ld.val_dim))                 
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    

def _pure_broadcast_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:    
    print("input", lhs, rhs)                                         
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

        ### Add padding
        lhs, rhs = _pad_tensors(lhs, rhs)
                
        print("lhs", lhs, lhs.val.shape)
        print("rhs", rhs, rhs.val.shape)
            
        assert _checkify_broadcast_compatibility(lhs.val, rhs.val), f"Shapes {lhs.val.shape} and {rhs.val.shape} not compatible for broadcast multiplication!"
        new_val = lhs.val * rhs.val
        out = _get_output_tensor(lhs, rhs, new_val)
        print("out", out, out.val.shape)
        res = _swap_back_axes(out)
        print("res", res, res.val.shape)
        return res


def _pure_dot_product_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """
    TODO docstring

    Args:
        lhs (SparseTensor): _description_
        rhs (SparseTensor): _description_

    Returns:
        SparseTensor: _description_
    """
    # If we do only contractions, no introduction of additional dimensions is
    # necessary
    lcontracting_axes, rcontracting_axes = [], []
    new_out_dims = lhs.out_dims
    l = len(lhs.out_dims)
    new_primal_dims = [DenseDimension(d.id, d.size, l+i) for i, d in enumerate(rhs.primal_dims)]        

    # Handling contracting variables
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            # In this case we have a contraction
            lcontracting_axes.append(ld.val_dim)
            rcontracting_axes.append(rd.val_dim)
                
    # Do the math using dot_general
    dimension_numbers = (tuple(lcontracting_axes), tuple(rcontracting_axes))
    dimension_numbers = (dimension_numbers, ((), ()))
    new_val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
    
    return SparseTensor(new_out_dims, new_primal_dims, new_val)


def _mixed_mul(lhs: SparseTensor, rhs: SparseTensor) -> SparseTensor:
    """_summary_

    Args:
        lhs (SparseTensor): _description_
        rhs (SparseTensor): _description_

    Returns:
        SparseTensor: _description_
    """
    new_out_dims = []
    new_primal_dims = []
    print("mixed_mul")
    print("lhs", lhs)
    print("rhs", rhs)
    l, r = len(lhs.out_dims), len(rhs.out_dims)
    lcontracting_axes, rcontracting_axes = [], []
    
    # We do contractions first
    for d in lhs.out_dims:
        if type(d) is DenseDimension:
            new_out_dims.append(DenseDimension(d.id, d.size, d.val_dim))
    
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension and type(rd) is DenseDimension:
            # In this case we have a contraction
            lcontracting_axes.append(ld.val_dim)
            rcontracting_axes.append(rd.val_dim)
                
    print(lcontracting_axes, rcontracting_axes)
    dimension_numbers = (tuple(lcontracting_axes), tuple(rcontracting_axes))
    dimension_numbers = (dimension_numbers, ((), ()))
    val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
    print(val.shape)
    
    lbroadcasting_axes, rbroadcasting_axes = [], []
    pos = []
    # Then we do broadcasting
    for (ld, rd) in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is SparseDimension or type(rd) is SparseDimension:
            # Here, we have a broadcasting over two tensors that are not just
            # Kronecker deltas
            if ld.val_dim is not None and rd.val_dim is not None:
                lval_dim = ld.val_dim - sum([1 for lc in lcontracting_axes if lc < ld.val_dim])
                rval_dim = ld.val_dim - sum([1 for rc in rcontracting_axes if rc < rd.val_dim]) + len(lhs.out_dims)
                lbroadcasting_axes.append(lval_dim)
                rbroadcasting_axes.append(rval_dim)
                pos.append(lval_dim)
                # The following cases cover ...
                if type(rd) is DenseDimension:
                    new_out_dims.insert(ld.other_id, DenseDimension(ld.other_id, ld.size, lval_dim))
                elif type(ld) is DenseDimension:
                    new_primal_dims.insert(rd.other_id-r, DenseDimension(rd.other_id-r, ld.size, rval_dim))
                else:
                    # TODO fix the val_dim here
                    new_out_dims.insert(ld.other_id, SparseDimension(ld.other_id, ld.size, lval_dim, rd.other_id-len(rbroadcasting_axes)))
                    new_primal_dims.insert(rd.other_id-r, SparseDimension(rd.other_id-len(rbroadcasting_axes), ld.size, lval_dim, ld.other_id))
            else:
                # In this case, one of the two tensors we contract is just a 
                # Kronecker delta so we can spare ourselves the contraction
                # and just reflag the dimension of the new tensor
                
                # The following cases cover ...
                if type(rd) is DenseDimension:
                    new_out_dims.append(DenseDimension(ld.id, ld.size, rd.val_dim))
                elif type(ld) is DenseDimension:
                    new_primal_dims.append(DenseDimension(rd.other_id, ld.size, ld.val_dim))
                else:
                    # TODO fix the val_dim here
                    val_dim = None
                    if ld.val_dim is not None:
                        val_dim = ld.val_dim
                    elif rd.val_dim is not None:
                        val_dim = rd.val_dim
                    new_out_dims.insert(ld.other_id, SparseDimension(ld.other_id, ld.size, val_dim, rd.other_id-len(rbroadcasting_axes)))
                    new_primal_dims.insert(rd.other_id-r, SparseDimension(rd.other_id-len(rbroadcasting_axes), ld.size, val_dim, ld.other_id))
    
    print(lbroadcasting_axes, rbroadcasting_axes)
    for (lb, rb, p) in zip(lbroadcasting_axes, rbroadcasting_axes, pos):
        val = jnp.diagonal(val, axis1=lb, axis2=rb)
        val = jnp.swapaxes(val, -1, p) # TODO do the swap_axes with a single transpose!

    print("diag", val.shape)
    
    delta = len(lcontracting_axes)
    for d in rhs.primal_dims:
        if type(d) is DenseDimension:
            # TODO this needs some adjustment in the val_dim property as well
            # since val_dim might change due to contractions etc.
            new_primal_dims.insert(d.id-l, DenseDimension(d.id, d.size, d.val_dim-delta))
    
    print("res", SparseTensor(new_out_dims, new_primal_dims, val))
    return SparseTensor(new_out_dims, new_primal_dims, val)


# TODO simplify this
def _sparse_add(lhs: SparseTensor, rhs: SparseTensor):
    """_summary_

    Args:
        lhs (SparseTensor): _description_
        rhs (SparseTensor): _description_

    Returns:
        _type_: _description_
    """
    assert lhs.shape == rhs.shape, f"Incompatible shapes {lhs.shape} and {rhs.shape}!"
    ldims, rdims = [], []
    new_out_dims, new_primal_dims = [], []
    _lshape, _rshape = [], [] 
    count = 0
    
    # Check the dimensionality of the 'out_dims' of both tensors
    for i, (ld, rd) in enumerate(zip(lhs.out_dims, rhs.out_dims)):
        if ld.val_dim is None and rd.val_dim is None:
            dim = None
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
            new_out_dims.append(SparseDimension(i, ld.size, dim, ld.other_id))
            _lshape.append(1)
            _rshape.append(1)
        else:
            if type(ld) is SparseDimension :
                _lshape.append(ld.size)
            elif type(rd) is SparseDimension:
                _rshape.append(ld.size)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_out_dims.append(DenseDimension(i, ld.size, dim))

    # Check the dimensionality of the 'primal_dims' of both tensors        
    for i, (ld, rd) in enumerate(zip(lhs.primal_dims, rhs.primal_dims), start=len(new_out_dims)):                 
        if type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id == rd.other_id:
            dim = new_out_dims[ld.other_id].val_dim
            new_primal_dims.append(SparseDimension(i, ld.size, dim, ld.other_id))
            _lshape.append(1)
            _rshape.append(1)
        else:            
            if ld.val_dim is None and rd.val_dim is None:
                dim = None
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
            
            if type(ld) is SparseDimension:
                _lshape.append(ld.size)
            elif type(rd) is SparseDimension:
                _rshape.append(rd.size)
            else:
                _lshape.append(1)
                _rshape.append(1)
            new_primal_dims.append(DenseDimension(i, ld.size, dim))
            
    lhs_val = lhs.materialize_dimensions(ldims)
    rhs_val = rhs.materialize_dimensions(rdims)
        
    # We need to materialize sparse dimensions for addition
    if len(_lshape) != 0:
        iota = jnp.bool_(eye_like(_lshape, len(lhs.out_dims)))
        lhs_val = jnp.where(iota, lhs_val, 0.)
    if len(_rshape) != 0:
        iota = jnp.bool_(eye_like(_rshape, len(rhs.out_dims)))
        rhs = jnp.where(iota, rhs_val, 0.)
    
    val = lhs_val + rhs_val
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    
    