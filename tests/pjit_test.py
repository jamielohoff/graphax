import unittest

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.sparse.utils import count_muls, count_muls_jaxpr

class PJITTest(unittest.TestCase): 
    def test_simple_pjit(self):
        def simple_pjit(x, y):
            x = jnp.sin(x)
            var = jnp.var(x)
            return y / var
        
        # def selecttest(y, z):
        #     x = jnp.array([2, 0, 1])
        #     y = jnp.sin(x)
        #     z = jnp.tanh(z)
        #     w = lax.select_n(x, y, z)
        #     return 2.0 * w
        
        # arr2 = jnp.array([1., 2., 3.])
        # arr3 = jnp.array([.1, .2, .3])
        # print(jax.jacrev(selecttest)(arr2, arr3))

        x = jnp.arange(0, 6) * 0.1
        y = jnp.ones((6,))
        print(jax.make_jaxpr(simple_pjit)(x, y))
        print(jax.make_jaxpr(jacve(simple_pjit, order="fwd", argnums=(0, 1)))(x, y))
        # jac_rev = jax.jit(jacve(simple_pjit, order="fwd", argnums=(0, 1)))
        # veres = jac_rev(x, y)

        # print(jax.make_jaxpr(jax.jacfwd(simple_pjit, argnums=(0, 1)))(x, y))
        # jax_jac_rev = jax.jit(jax.jacfwd((simple_pjit), argnums=(0, 1)))
        # revres = jax_jac_rev(x, y)

        # print(veres[0])
        # print(revres[0])
        # self.assertTrue(tree_allclose(veres, revres))
        
if __name__ == '__main__':
    unittest.main()

