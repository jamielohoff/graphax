from typing import Any, Callable, Sequence, Generator
from chex import Array

import jax
import jax.numpy as jnp
from tensor import SparseTensor

# TODO: make parent class, or inherit sparse tensor ??

@dataclass
class DenseDimension:
    id: int
    size: int
    val_dim: int | None

@dataclass
class SparseDimension:
    id: int
    size: int
    val_dim: int
    other_id: int

Dimension = DenseDimension | SparseDimension

class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    shape: Sequence[int]
    blocks: Sequence[Array]
    pre_transforms: Sequence[Callable] 
    post_transforms: Sequence[Callable]

    def __init__(self,
                 out_dims: Sequence[Dimension],
                 primal_dims: Sequence[Dimension],
                 blocks: Sequence[Array],
                 pre_transforms: Sequence[Callable] = None,
                 post_transforms: Sequence[Callable] = None) -> None:

        if pre_transforms is None:
            pre_transforms = []
        if post_transforms is None:
            post_transforms = []

        self.out_dims = out_dims if isinstance(out_dims, tuple) else tuple(out_dims)
        self.primal_dims = primal_dims if isinstance(primal_dims, tuple) else tuple(primal_dims)

        self.out_shape = [d.size for d in out_dims]
        self.primal_shape = [d.size for d in primal_dims]

        self.shape = tuple(self.out_shape + self.primal_shape)

        self.blocks = blocks
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms

    

