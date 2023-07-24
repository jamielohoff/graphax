from typing import Sequence, Tuple, Union, NamedTuple
from functools import reduce

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import ShapedArray


def _eye_like(shape, out_len):
    primal_shape = shape[out_len:]
    out_shape = shape[:out_len]
    if any([primal_shape == out_shape]):
        primal_size = reduce((lambda x, y: x*y), primal_shape)
        out_size = reduce((lambda x, y: x*y), out_shape)
        return jnp.eye(out_size, primal_size).reshape(*out_shape, *primal_shape)
    else:
        l = len(out_shape)
        val = 1.
        for i, o in enumerate(out_shape):
            j = primal_shape.index(o)
            _shape = [1]*len(shape)
            _shape[i] = o
            _shape[l+j] = o
            kronecker = jnp.eye(o).reshape(_shape)
            val *= kronecker
        return val
            

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
        print("dims", dims)
        _shape = list(self.val.shape)
        _broadcast_dims = []
        for d in dims:
            _shape.insert(d, 1)
            
        # print("_shape", _shape)
        total_shift = 0
        for i, vd in enumerate(self.val.shape):
            shift = 0
            for d in dims:
                if d <= i + total_shift:
                    shift += 1
            _broadcast_dims.append(i+shift)
            total_shift += shift
        print("broadcast", self, _shape, _broadcast_dims)
        return self.val.reshape(_shape) # lax.broadcast_in_dim(self.val, _shape, _broadcast_dims)
    
    # TODO simplify this
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
            # TODO this is just a workaround!!!!
            # Squeeze everything is a bad idea
            # print(self.out_dims, self.primal_dims)
            # print(self.shape, self.val.shape, _shape, _sparse_shape, _broadcast_dims)
            val = self.val.reshape(_shape) # lax.broadcast_in_dim(self.val, _shape, _broadcast_dims)
            return val*_eye_like(_sparse_shape, len(self.out_dims))
    
    
# TODO simplify this
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
    
    # print(lhs, rhs)
    # print("error", _lshape, len(lhs.out_dims), _rshape, len(rhs.out_dims))
    
    # We need to materialize sparse dimensions for addition
    if len(_lshape) != 0:
        lhs_val *= _eye_like(_lshape, len(lhs.out_dims))
    if len(_rshape) != 0:
        rhs_val *= _eye_like(_rshape, len(rhs.out_dims))
    
    val = lhs_val + rhs_val
    
    return SparseTensor(new_out_dims, new_primal_dims, val)
    

def _mul(lhs: SparseTensor, rhs: SparseTensor):
    # TODO add docstring
    lcontracting_dims, rcontracting_dims = [], []
    new_out_dims, new_primal_dims = [], []
    ldims, rdims = [], []
        
    dim_count = 0
    l = len(lhs.out_dims)
    r = len(rhs.out_dims)
    # Handling dependent variables
    for ld in lhs.out_dims:
        if type(ld) is DenseDimension:
            rdims.append(ld.id)
            new_dim = DenseDimension(ld.id, ld.size, dim_count)
            new_out_dims.insert(ld.id, new_dim)
            dim_count += 1
            
        elif type(ld) is SparseDimension:
            rd = rhs.out_dims[ld.other_id-l]
            if type(rd) is DenseDimension:
                # lhs contains a Kronecker delta
                # Adds DenseDimension to out_dims
                new_dim = DenseDimension(ld.id, ld.size, dim_count)
                new_out_dims.insert(ld.id, new_dim)
                if ld.val_dim is None:
                    ldims.append(ld.id)
                dim_count += 1
                
            elif type(rd) is SparseDimension:
                # Both dimension contain a Kronecker delta
                if ld.val_dim is None and rd.val_dim is None:
                    dim = None
                elif rd.val_dim is None:
                    rdims.append(ld.id)
                    dim = dim_count
                    dim_count += 1
                elif ld.val_dim is None:
                    ldims.append(rd.id)
                    dim = dim_count
                    dim_count += 1
                else:
                    dim = dim_count
                    dim_count += 1

                new_out_dim = SparseDimension(ld.id, ld.size, dim, rd.other_id)
                new_out_dims.insert(ld.id, new_out_dim)
                new_primal_dim = SparseDimension(rd.other_id, rd.size, dim, ld.id)
                new_primal_dims.insert(rd.other_id-r, new_primal_dim)                
            else:
                ValueError(str(rd) + " is not a valid dimension type!")
        else:
            ValueError(str(ld) + " is not a valid dimension type!")
        
    
    # Handling contracting variables
    for ld, rd in zip(lhs.primal_dims, rhs.out_dims):
        if type(ld) is DenseDimension:
            if type(rd) is DenseDimension:
                # In this case we have a contraction
                lcontracting_dims.append(ld.val_dim)
                rcontracting_dims.append(rd.val_dim)
            else:
                ValueError(str(rd) + " is not a valid dimension type!")
        else:
            ValueError(str(ld) + " is not a valid dimension type!")
    
    # Handling independent variables
    for rd in rhs.primal_dims:
        if type(rd) is DenseDimension:
            ldims.append(rd.id)
            new_dim = DenseDimension(rd.id, rd.size, dim_count)
            new_primal_dims.insert(rd.id-l, new_dim)  
            dim_count += 1
        elif type(rd) is SparseDimension:
            ld = lhs.primal_dims[rd.other_id]
            if type(ld) is DenseDimension:
                # rhs contains a Kronecker delta
                new_dim = DenseDimension(rd.id, rd.size, dim_count)
                new_primal_dims.insert(rd.id-l, new_dim)
                if rd.val_dim is None:
                    rdims.append(rd.other_id)
                dim_count += 1
            else:
                ValueError(str(ld) + " is not a valid dimension type!")
        else:
            ValueError(str(rd) + " is not a valid dimension type!")
    
    # Executing the appropriate computations
    if len(lcontracting_dims) > 0:
        # If we do contractions first, the process looks different
        print("we fail here!")

        dimension_numbers = (tuple(lcontracting_dims), tuple(rcontracting_dims))
        dimension_numbers = (dimension_numbers, ((), ()))
        print(lhs.val.shape, rhs.val.shape, dimension_numbers)
        val = lax.dot_general(lhs.val, rhs.val, dimension_numbers)
        
        print(val.shape)
        
        if len(ldims) > 0 or len(rdims) > 0:
            pass
        else:
            pass
    else: 
        lhs_val = lhs.materialize_dimensions(ldims)
        rhs_val = rhs.materialize_dimensions(rdims)            
        val = lhs_val * rhs_val
            
    return SparseTensor(new_out_dims, new_primal_dims, val)

    