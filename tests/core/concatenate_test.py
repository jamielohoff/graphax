import unittest

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


class ConcatenateTest(unittest.TestCase):
    def test_two_concatenates_same_scalar(self):
        """Reproduces shape mismatch when the same scalar feeds two concatenate ops."""

        def fn(v):
            pl = jnp.sum(v, keepdims=True)  # [1]

            _p = jnp.concatenate([pl, jnp.zeros(2, dtype=jnp.float32)])  # [3]
            dF = jnp.concatenate([pl, pl, pl])  # [3]

            return _p - dF  # [3]

        v = jnp.array([0.1, 0.2, 0.3])

        jacve_f = jax.jit(jacve(fn, order="rev", argnums=(0,)))
        veres = jacve_f(v)

        jacrev_f = jax.jit(jax.jacrev(fn, argnums=(0,)))
        revres = jacrev_f(v)

        self.assertTrue(tree_allclose(veres, revres))


if __name__ == "__main__":
    unittest.main()
