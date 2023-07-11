from typing import Sequence, Tuple, Union, NamedTuple
from functools import reduce

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import ShapedArray


def _eye_like(shape, out_len):
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    primal_size = reduce((lambda x, y: x * y), primal_shape)
    out_size = reduce((lambda x, y: x * y), out_shape)
    return jnp.eye(out_size, primal_size).reshape(*out_shape, *primal_shape)


class DenseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int

class SparseDimension(NamedTuple):
    id: int
    size: int
    val_dim: int
    other_id: int

Dimension = Union[DenseDimension, SparseDimension]
        

class SparseTensor:
    """
    TODO add docstring
    
    if out_dims or primal_dims is empty, this implies a scalar dependent or
    independent variable. 
    if both are empty, then we have a scalar value and everything becomes trivial
    and the `val` field contains the value of the singleton partial
    """
    out_dims: Sequence[Dimension]
    primal_dims: Sequence[Dimension]
    shape: Sequence[int] # True shape of the tensor
    val: ShapedArray
    # NOTE: We always assume that the dimensions are ordered in ascending order
    
    def __init__(self, out_dims, primal_dims, val) -> None:        
        self.out_dims = out_dims if type(out_dims) is Tuple else tuple(out_dims)
        self.primal_dims = primal_dims if type(primal_dims) is Tuple else tuple(primal_dims)
        out_shape = [d.size for d in out_dims]
        primal_shape = [d.size for d in primal_dims]
        self.shape = out_shape + primal_shape
        self.val = val
            
    def __repr__(self) -> str:
        return "SparseTensor: \n" \
                "   shape = " + str(self.shape) + "\n" \
                "   out_dims = " + str(self.out_dims) + "\n" \
                "   primal_dims = " + str(self.primal_dims) + "\n" \
                "   val = " + str(self.val)  
                
    def __add__(self, _tensor):
        return _add(self, _tensor)
    
    def __mul__(self, _tensor):
        return _mul(self, _tensor)
        
    def materialize_dimensions(self, dims):
        # TODO add docstring
        dims = sorted(dims)
        _shape = list(self.val.shape)
        _broadcast_dims = []
        for d in dims:
            _shape.insert(d, 1)
            
        for i, vd in enumerate(self.val.shape):
            shift = sum([1 for d in dims if d <= i])
            _broadcast_dims.append(i+shift)
        return lax.broadcast_in_dim(self.val, _shape, _broadcast_dims)
    
    def materialize_actual_shape(self):
        # Materializes tensor to actual shape where Dimensions with val_dim=None
        # are materialized as 1     
        _sparse_shape = []
        _shape, _broadcast_dims = [], []
        for i, d in enumerate(self.out_dims):
            if d.val_dim is not None:
                _shape.append(d.size)
                _broadcast_dims.append(i)
            else:
                _shape.append(1)
                
            if type(d) is SparseDimension:
                _sparse_shape.append(d.size)
            else:
                _sparse_shape.append(1)
                    
        for i, d in enumerate(self.primal_dims, start=len(self.out_dims)):
            if d.val_dim is not None:
                if type(d) is DenseDimension:
                    _shape.append(d.size)
                    _broadcast_dims.append(i)
                else:
                    _shape.append(1)
            else:
                _shape.append(1)

            if type(d) is SparseDimension:
                _sparse_shape.append(d.size)
            else:
                _sparse_shape.append(1)
                
        if len(self.out_dims) == 0 or len(self.primal_dims) == 0:
            return self.val
        else:
            val = lax.broadcast_in_dim(self.val, _shape, _broadcast_dims)
            return val*_eye_like(_sparse_shape, len(self.out_dims))
    
    
def _add(lhs: SparseTensor, rhs: SparseTensor):
    assert lhs.shape == rhs.shape
    # Here we just do broadcasting
    ldims, rdims = [], []
    new_out_dims, new_primal_dims = [], []
    _lshape, _rshape = [], []
    
    count = 0
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

            
    for i, (ld, rd) in enumerate(zip(lhs.primal_dims, rhs.primal_dims), start=len(new_out_dims)):                 
        if type(ld) is SparseDimension and type(rd) is SparseDimension and ld.other_id == rd.other_id:
            dim = new_out_dims[ld.other_id].val_dim
            new_primal_dims.append(SparseDimension(i, ld.size, dim, ld.other_id))
            _lshape.append(1)
            _rshape.append(1)
        else:
            new_primal_dims.append(DenseDimension(i, ld.size, dim))
            
            if type(ld) is SparseDimension:
                _lshape.append(ld.size)
            elif type(rd) is SparseDimension:
                _rshape.append(rd.size)
            else:
                _lshape.append(1)
                _rshape.append(1)
            
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
            
    lhs_val = lhs.materialize_dimensions(ldims)
    rhs_val = rhs.materialize_dimensions(rdims)
    
    # We need to materialize sparse dimensions for addition
    if len(_lshape) != 0:
        lhs_val *= _eye_like(_lshape, len(lhs.out_dims))
    if len(_rshape) != 0:
        rhs_val *= _eye_like(_rshape, len(rhs.out_dims))
    
    val = lhs_val + rhs_val
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    
    
def _mul(lhs: SparseTensor, rhs: SparseTensor):
    # TODO add docstring
    lbroadcasting_dims, rbroadcasting_dims = [], []
    lcontracting_dims, rcontracting_dims = [], []
    new_out_dims, new_primal_dims = [], []
    ldims, rdims = [], []
        
    dim_count, lcount, rcount = 0, 0, 0
    l = len(lhs.out_dims)
    
    # Handling dependent variables
    for ld in lhs.out_dims:
        if type(ld) is DenseDimension:
            rdims.append(0)
            new_dim = DenseDimension(lcount, ld.size, dim_count)
            new_out_dims.append(new_dim)   
            lcount += 1       
            dim_count += 1
    
    # Handling contracting variables and adjusting sparsity map
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension:
            if type(rd) is DenseDimension:
                # In this case we have a contraction
                lcontracting_dims.append(ld.val_dim) # (ld.id, ld.val_dim)
                rcontracting_dims.append(rd.val_dim) # (rd.id, rd.val_dim)
                
            elif type(rd) is SparseDimension:
                # rhs contains a Kronecker delta
                # Adds DenseDimension to primal_dims
                rbroadcasting_dims.append((ld.id, ld.val_dim))
                new_dim = DenseDimension(rcount, rd.size, dim_count)
                new_primal_dims.append(new_dim)
                if rd.val_dim is None:
                    rdims.append(dim_count)
                rcount += 1
                dim_count += 1
            else:
                ValueError(rd + " is not a valid dimension type!")
                
        elif type(ld) is SparseDimension:
            if type(rd) is DenseDimension:
                # lhs contains a Kronecker delta
                # Adds DenseDimension to out_dims
                lbroadcasting_dims.append((rd.id, rd.val_dim))
                new_dim = DenseDimension(lcount, ld.size, dim_count)
                new_out_dims.append(new_dim)
                
                if ld.val_dim is None:
                    ldims.append(dim_count)
                lcount += 1
                dim_count += 1
                
            elif type(rd) is SparseDimension:
                # Both dimension contain a Kronecker delta
                if rd.val_dim is None and ld.val_dim is None:
                    dim = None
                elif rd.val_dim is None:
                    rdims.append(dim_count)
                    dim = dim_count
                    dim_count += 1
                elif ld.val_dim is None:
                    ldims.append(dim_count)
                    dim = dim_count
                    dim_count += 1
                else:
                    dim = dim_count
                    dim_count += 1
                new_out_dim = SparseDimension(lcount, ld.size, dim, l+rcount)
                new_out_dims.append(new_out_dim)
                new_primal_dim = SparseDimension(l+rcount, rd.size, dim, lcount)
                new_primal_dims.append(new_primal_dim)                
                lcount += 1
                rcount += 1   
            else:
                ValueError(rd + " is not a valid dimension type!")
        else:
            ValueError(ld + " is not a valid dimension type!")
    
    # Handling independent variables
    for i, rd in enumerate(rhs.primal_dims):
        if type(rd) is DenseDimension:
            ldims.append(lhs.val.ndim)
            new_dim = DenseDimension(l+rcount, rd.size, dim_count)
            new_primal_dims.insert(i, new_dim)   
            rcount += 1  
            dim_count += 1
    
    # Executing the appropriate computations
    if len(lcontracting_dims) > 0:
        # If we do contractions first, the process looks different
        print("we fail here!")

        dimension_numbers = (tuple(lcontracting_dims), tuple(rcontracting_dims))
        dimension_numbers = (dimension_numbers, ((), ()))
        print(lhs.val.shape, rhs.val.shape, dimension_numbers)
        val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
        
        print(val.shape)
        
        if len(lbroadcasting_dims) > 0 or len(rbroadcasting_dims) > 0:
            pass
        else:
            pass
    else: 
        lhs_val = lhs.materialize_dimensions(ldims)
        rhs_val = rhs.materialize_dimensions(rdims)            
        val = lhs_val * rhs_val
    
    return SparseTensor(new_out_dims, new_primal_dims, val)

    