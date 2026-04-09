import unittest
from functools import partial
from typing import Callable, Sequence

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand


from graphax import jacve, tree_allclose
from example_test import test_fwd, test_rev, test_order


class HessianTests(unittest.TestCase): 
    def test_NeuralNetworkHessian(self):
        batchsize = 16
        @partial(jax.vmap, in_axes=(0, None, None, None, None, 0))
        def NeuralNetwork(x, W1, b1, W2, b2, y):
            y1 = W1 @ x
            z1 = y1 + b1
            a1 = jnp.tanh(z1)
            
            y2 = W2 @ a1
            z2 = y2 + b2
            return 0.5*(jnp.tanh(z2) - y)**2
        
        def f(x, W1, b1, W2, b2, y):
            out = NeuralNetwork(x, W1, b1, W2, b2, y)
            return out.sum()
            
        key = jrand.PRNGKey(42)

        x = jnp.ones((batchsize, 4))
        y = jrand.normal(key, (batchsize, 4))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))
        args = (x, W1, b1, W2, b2, y)
        argnums = (1, 2, 3, 4)
        
        jac_rev_fn = jacve(f, order="rev", argnums=argnums)

        self.assertTrue(test_fwd(jac_rev_fn, argnums, *args))
        self.assertTrue(test_rev(jac_rev_fn, argnums, *args))


if __name__ == "__main__":
    unittest.main()

