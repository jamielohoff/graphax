import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.sparse.utils import count_muls, count_muls_jaxpr

class PJITTest(unittest.TestCase): 
    def test_simple_pjit(self):
        def pjit_with_var(x, y):
            x = jnp.sin(x)
            var = jnp.var(x)
            return y / var

        x = jnp.arange(0, 6) * 0.1
        y = jnp.ones((6,))
        print(jax.make_jaxpr(pjit_with_var)(x, y))
        print(jax.make_jaxpr(jacve(pjit_with_var, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(pjit_with_var, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(pjit_with_var, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd((pjit_with_var), argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        print(veres[1])
        print(revres[1])
        self.assertTrue(tree_allclose(veres, revres))
        
if __name__ == '__main__':
    unittest.main()