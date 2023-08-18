import jax
import jax.numpy as jnp
import jax.tree_util as jtu

def tree_allclose(tree1, tree2):
    is_equal = jtu.tree_map(jnp.allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


def zeros_like(invar, outvar):
    in_shape = invar.aval.shape
    out_shape = outvar.aval.shape
    
    if in_shape == () and out_shape == ():
        return 0.
    else:
        shape = (*in_shape, *out_shape)
        return jnp.zeros_like(shape)

