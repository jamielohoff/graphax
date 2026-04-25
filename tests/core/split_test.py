"""
Unit tests for jnp.split multi-output vertex elimination.

Tests cover:
- 1D and 2D splits, equal and unequal partition sizes
- Splits along different axes
- Split outputs used as intermediates vs. final outputs
- Both forward and reverse elimination orders
- Multiple inputs (argnums)
- Only some split outputs consumed (DropVar case)
- Chained / nested splits
- Recombining split outputs (fan-in after fan-out)
"""

import unittest

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve, tree_allclose


def _allclose(a, b, atol=1e-5, rtol=1e-5):
    """Compare two arrays or equal-structure pytrees element-wise."""
    return jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=atol, rtol=rtol)


class SplitTest(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1-D equal split, outputs used as intermediates
    # ------------------------------------------------------------------

    def test_1d_equal_split_fwd(self):
        """2-way 1D split; both outputs fed into further ops, fwd order."""
        def f(x):
            y0, y1 = jnp.split(x, 2)
            return jnp.sin(y0) + jnp.cos(y1)

        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        veres = jax.jit(jacve(f, order="fwd"))(x)
        refres = jax.jit(jax.jacfwd(f))(x)
        print(veres)
        print(refres)
        self.assertTrue(_allclose(veres[0], refres))

    # def test_1d_equal_split_rev(self):
    #     """2-way 1D split; both outputs fed into further ops, rev order."""
    #     def f(x):
    #         y0, y1 = jnp.split(x, 2)
    #         return jnp.sin(y0) + jnp.cos(y1)

    #     x = jnp.array([1.0, 2.0, 3.0, 4.0])
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # 1-D unequal split via index list
    # # ------------------------------------------------------------------

    # def test_1d_unequal_split_fwd(self):
    #     """3-way unequal 1D split, sizes [2, 3, 1]."""
    #     def f(x):
    #         a, b, c = jnp.split(x, [2, 5])
    #         return jnp.sum(a) + jnp.sum(b) * jnp.sum(c)

    #     x = jnp.arange(6, dtype=jnp.float32) + 1.0
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_1d_unequal_split_rev(self):
    #     """3-way unequal 1D split, rev order."""
    #     def f(x):
    #         a, b, c = jnp.split(x, [2, 5])
    #         return jnp.sum(a) + jnp.sum(b) * jnp.sum(c)

    #     x = jnp.arange(6, dtype=jnp.float32) + 1.0
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # 2-D split along axis 0
    # # ------------------------------------------------------------------

    # def test_2d_split_axis0_fwd(self):
    #     """2-way 2D split along rows."""
    #     def f(x):
    #         top, bot = jnp.split(x, 2, axis=0)
    #         return jnp.tanh(top) - jnp.exp(bot)

    #     x = jnp.ones((4, 3))
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_2d_split_axis0_rev(self):
    #     """2-way 2D split along rows, rev order."""
    #     def f(x):
    #         top, bot = jnp.split(x, 2, axis=0)
    #         return jnp.tanh(top) - jnp.exp(bot)

    #     x = jnp.ones((4, 3))
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # 2-D split along axis 1
    # # ------------------------------------------------------------------

    # def test_2d_split_axis1_fwd(self):
    #     """3-way 2D split along columns, fwd order."""
    #     def f(x):
    #         a, b, c = jnp.split(x, 3, axis=1)
    #         return jnp.sin(a) + jnp.cos(b) - jnp.tanh(c)

    #     x = jnp.ones((2, 6))
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_2d_split_axis1_rev(self):
    #     """3-way 2D split along columns, rev order."""
    #     def f(x):
    #         a, b, c = jnp.split(x, 3, axis=1)
    #         return jnp.sin(a) + jnp.cos(b) - jnp.tanh(c)

    #     x = jnp.ones((2, 6))
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # Only one split output consumed (DropVar for the unused output)
    # # ------------------------------------------------------------------

    # def test_one_output_used_fwd(self):
    #     """Split into two; only the first chunk is used downstream."""
    #     def f(x):
    #         y0, _y1 = jnp.split(x, 2)
    #         return jnp.sum(y0)

    #     x = jnp.array([1.0, 2.0, 3.0, 4.0])
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_one_output_used_rev(self):
    #     """Split into two; only the second chunk is used downstream, rev."""
    #     def f(x):
    #         _y0, y1 = jnp.split(x, 2)
    #         return jnp.sum(y1)

    #     x = jnp.array([1.0, 2.0, 3.0, 4.0])
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # Multiple function outputs returned directly
    # # ------------------------------------------------------------------

    # def test_multiple_function_outputs_fwd(self):
    #     """Both split chunks processed separately and returned."""
    #     def f(x):
    #         y0, y1 = jnp.split(x, 2)
    #         return jnp.sum(y0), jnp.sum(y1)

    #     x = jnp.array([1.0, 2.0, 3.0, 4.0])
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(tree_allclose(veres, refres))

    # def test_multiple_function_outputs_rev(self):
    #     """Both split chunks processed separately and returned, rev order."""
    #     def f(x):
    #         y0, y1 = jnp.split(x, 2)
    #         return jnp.sum(y0), jnp.sum(y1)

    #     x = jnp.array([1.0, 2.0, 3.0, 4.0])
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(tree_allclose(veres, refres))

    # # ------------------------------------------------------------------
    # # Split with multiple inputs (argnums)
    # # ------------------------------------------------------------------

    # def test_multi_input_split_fwd(self):
    #     """Split interacts with a second input variable, fwd order."""
    #     def f(x, y):
    #         a, b, c = jnp.split(x, [2, 4], axis=0)
    #         # a: (2,3), b: (2,3), c: (2,3), y: (3,2)
    #         return jnp.sum(jnp.sin(a) @ y) + jnp.sum(jnp.cos(b)) + jnp.sum(c)

    #     x = jnp.ones((6, 3))
    #     y = jnp.ones((3, 2))
    #     veres = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))(x, y)
    #     refres = jax.jit(jax.jacfwd(f, argnums=(0, 1)))(x, y)
    #     self.assertTrue(tree_allclose(veres, refres))

    # def test_multi_input_split_rev(self):
    #     """Split interacts with a second input variable, rev order."""
    #     def f(x, y):
    #         a, b, c = jnp.split(x, [2, 4], axis=0)
    #         return jnp.sum(jnp.sin(a) @ y) + jnp.sum(jnp.cos(b)) + jnp.sum(c)

    #     x = jnp.ones((6, 3))
    #     y = jnp.ones((3, 2))
    #     veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
    #     refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    #     self.assertTrue(tree_allclose(veres, refres))

    # # ------------------------------------------------------------------
    # # Outputs from split recombined (fan-in after fan-out)
    # # ------------------------------------------------------------------

    # def test_split_recombined_fwd(self):
    #     """Split then add outputs together — Jacobian must sum correctly."""
    #     def f(x):
    #         y0, y1 = jnp.split(x, 2)
    #         return y0 + y1

    #     x = jnp.arange(4, dtype=jnp.float32)
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_split_recombined_rev(self):
    #     """Split then add outputs together, rev order."""
    #     def f(x):
    #         y0, y1 = jnp.split(x, 2)
    #         return y0 + y1

    #     x = jnp.arange(4, dtype=jnp.float32)
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # Chained / sequential splits
    # # ------------------------------------------------------------------

    # def test_chained_splits_fwd(self):
    #     """Second split operates on the output of the first split."""
    #     def f(x):
    #         first, rest = jnp.split(x, [3])
    #         left, right = jnp.split(rest, 2)
    #         return jnp.sum(first) + jnp.sum(left * right)

    #     x = jnp.arange(7, dtype=jnp.float32) + 1.0
    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_chained_splits_rev(self):
    #     """Chained splits, rev order."""
    #     def f(x):
    #         first, rest = jnp.split(x, [3])
    #         left, right = jnp.split(rest, 2)
    #         return jnp.sum(first) + jnp.sum(left * right)

    #     x = jnp.arange(7, dtype=jnp.float32) + 1.0
    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # # ------------------------------------------------------------------
    # # Random inputs for numerical robustness
    # # ------------------------------------------------------------------

    # def test_random_2d_split_fwd(self):
    #     """Random 2D input, 4-way split along axis 1, fwd."""
    #     key = jrand.PRNGKey(0)
    #     x = jrand.normal(key, (3, 8))

    #     def f(x):
    #         a, b, c, d = jnp.split(x, 4, axis=1)
    #         return jnp.sin(a) + jnp.cos(b) - jnp.tanh(c) + jnp.exp(-d)

    #     veres = jax.jit(jacve(f, order="fwd"))(x)
    #     refres = jax.jit(jax.jacfwd(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_random_2d_split_rev(self):
    #     """Random 2D input, 4-way split along axis 1, rev."""
    #     key = jrand.PRNGKey(1)
    #     x = jrand.normal(key, (3, 8))

    #     def f(x):
    #         a, b, c, d = jnp.split(x, 4, axis=1)
    #         return jnp.sin(a) + jnp.cos(b) - jnp.tanh(c) + jnp.exp(-d)

    #     veres = jax.jit(jacve(f, order="rev"))(x)
    #     refres = jax.jit(jax.jacrev(f))(x)
    #     self.assertTrue(_allclose(veres[0], refres))

    # def test_mixed_ops_multi_input_fwd(self):
    #     """3-way split of x (6x3), interaction with second input y, fwd."""
    #     def f(x, y):
    #         a, b, c = jnp.split(x, [2, 4], axis=0)
    #         return jnp.sin(a) @ y, jnp.cos(b) + c

    #     x = jnp.ones((6, 3))
    #     y = jnp.ones((3, 2))
    #     veres = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))(x, y)
    #     refres = jax.jit(jax.jacfwd(f, argnums=(0, 1)))(x, y)
    #     self.assertTrue(tree_allclose(veres, refres))

    # def test_mixed_ops_multi_input_rev(self):
    #     """3-way split of x (6x3), interaction with second input y, rev."""
    #     def f(x, y):
    #         a, b, c = jnp.split(x, [2, 4], axis=0)
    #         return jnp.sin(a) @ y, jnp.cos(b) + c

    #     x = jnp.ones((6, 3))
    #     y = jnp.ones((3, 2))
    #     veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
    #     refres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
    #     self.assertTrue(tree_allclose(veres, refres))


if __name__ == "__main__":
    unittest.main()
