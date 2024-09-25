import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.sparse.utils import count_muls, count_muls_jaxpr

class MultiOutputTest(unittest.TestCase): 
    def test_split(self):
        def array_split(x, y):
            z, u, v = jnp.split(x, [2, 4], axis=0)
            a = jnp.sin(z) @ y
            b = u + jnp.log(v)
            return a, b

        x = jnp.ones((6, 3))
        y = jnp.ones((3, 2))
        print(jax.make_jaxpr(array_split)(x, y))
        print(jax.make_jaxpr(jacve(array_split, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(array_split, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(array_split, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd((array_split), argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        print(veres[1])
        print(revres[1])
        self.assertTrue(tree_allclose(veres, revres))
        
if __name__ == '__main__':
    unittest.main()
    
    