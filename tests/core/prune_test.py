import unittest

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose


class TestPruneGraph(unittest.TestCase):
    """Tests for the _prune_graph primitive that removes
    non-differentiated inputs and dead vertices from the
    computational graph before vertex elimination."""

    # --- Pruning non-differentiated inputs (argnums) ---

    def test_prune_single_argnum(self):
        """Only differentiate w.r.t. first arg; second arg should be pruned."""
        def f(x, y):
            return x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0,))
        result = jac_fn(x, y)

        # jacve unwraps single argnum: returns out[0] directly
        expected = jax.jacfwd(f, argnums=(0,))(x, y)[0]
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_second_argnum(self):
        """Only differentiate w.r.t. second arg; first arg should be pruned."""
        def f(x, y):
            return x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(1,))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(1,))(x, y)[0]
        self.assertTrue(tree_allclose(result, expected))

    # Note: argnums=() is not supported by jacve (_iota_shape fails on empty list)
    # so we skip that edge case here.

    def test_prune_all_argnums(self):
        """Differentiate w.r.t. all args — nothing pruned."""
        def f(x, y):
            return x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    # --- Pruning dead vertices (stop_gradient) ---

    def test_prune_dead_vertex_stop_grad_x(self):
        """stop_gradient on x creates a dead vertex for x in the graph."""
        def f(x, y):
            x_stopped = jax.lax.stop_gradient(x)
            return x_stopped * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_dead_vertex_stop_grad_y(self):
        """stop_gradient on y creates a dead vertex for y."""
        def f(x, y):
            y_stopped = jax.lax.stop_gradient(y)
            return x * y_stopped

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_dead_vertex_chain(self):
        """Multiple stop_gradient in a chain — multiple dead vertices."""
        def f(a, b, c):
            a_stopped = jax.lax.stop_gradient(a)
            b_stopped = jax.lax.stop_gradient(b)
            return a_stopped * b_stopped * c

        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        c = jnp.array([5.0, 6.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1, 2))
        result = jac_fn(a, b, c)

        expected = jax.jacfwd(f, argnums=(0, 1, 2))(a, b, c)
        self.assertTrue(tree_allclose(result, expected))

    # --- Pruning with constvars ---

    def test_prune_with_const(self):
        """Function with captured const — const should not be pruned incorrectly."""
        const = jnp.array([10.0, 20.0])

        def f(x, y):
            return x * y + const

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_with_const_single_argnum(self):
        """Const + single argnum — const must survive pruning."""
        const = jnp.array([10.0, 20.0])

        def f(x, y):
            return x * y + const

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0,))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0,))(x, y)[0]
        self.assertTrue(tree_allclose(result, expected))

    # --- Multi-output ---

    def test_prune_multi_output(self):
        """Multi-output function — pruning should not break Jacobian shape."""
        def f(x, y):
            return x * y, x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_multi_output_single_argnum(self):
        """Multi-output with single argnum."""
        def f(x, y):
            return x * y, x + y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0,))
        result = jac_fn(x, y)

        # Use argnums=0 (not (0,)) so jax.jacfwd returns (jac0, jac1) directly
        expected = jax.jacfwd(f, argnums=0)(x, y)
        self.assertTrue(tree_allclose(result[0], expected[0]))
        self.assertTrue(tree_allclose(result[1], expected[1]))

    # --- Larger / more complex graphs ---

    def test_prune_deep_graph(self):
        """Deep computation graph — pruning should still produce correct result."""
        def f(x, y, z):
            a = x * y
            b = a + z
            c = b * x
            d = jax.lax.stop_gradient(c)
            return d * y + z

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        z = jnp.array([5.0, 6.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1, 2))
        result = jac_fn(x, y, z)

        expected = jax.jacfwd(f, argnums=(0, 1, 2))(x, y, z)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_forward_order(self):
        """Pruning with forward elimination order."""
        def f(x, y):
            return x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="fwd", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))

    def test_prune_reverse_order(self):
        """Pruning with reverse elimination order."""
        def f(x, y):
            return x * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        jac_fn = jacve(f, order="rev", argnums=(0, 1))
        result = jac_fn(x, y)

        expected = jax.jacfwd(f, argnums=(0, 1))(x, y)
        self.assertTrue(tree_allclose(result, expected))


if __name__ == "__main__":
    unittest.main()
