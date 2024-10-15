from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


class BlockSparseTensor:
    out_dims: Any
    primal_dims: Any
    shape: Sequence[int]
    blocks: Sequence[jnp.ndarray]
    pre_transforms: Sequence[Callable] 
    post_transforms: Sequence[Callable]

    

