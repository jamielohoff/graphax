import unittest

import jax
import jax.numpy as jnp

from graphax import jacve, tree_allclose

class TestStopGradient(unittest.TestCase):
    def test_stop_gradient_flow(self):
        """Verify that stop_gradient correctly stops the gradient flow."""
        def f(x, y):
            # x should have zero gradient because of stop_gradient
            # y should have gradient
            x_stopped = jax.lax.stop_gradient(x)
            return x_stopped * y

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])

        # Using jacve to compute Jacobian
        jac_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = jac_fn(x, y)

        # Expected: grad wrt x is 0, grad wrt y is x
        # veres[0] is jac wrt x, veres[1] is jac wrt y
        # veres[0] should be zeros
        # veres[1] should be [1.0, 2.0] (for y, since f is x_stopped * y)

        # We check the actual values.
        # Since f is x*y (with x stopped), jac wrt x is 0, jac wrt y is x.
        # veres is [jac_x, jac_y]

        # Using jax.jacrev as reference
        jax_jac_rev = jax.jit(jax.jacrev(f, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        self.assertTrue(tree_allclose(veres, revres))

        # Explicitly check that gradient wrt x is zero
        self.assertTrue(jnp.all(veres[0] == 0.0))
        # Jacobian wrt y is diag(x): df_i/dy_j = x_i if i==j else 0
        self.assertTrue(jnp.all(jnp.diag(veres[1]) == x))

    def test_stop_gradient_chain(self):
        """Verify stop_gradient works in a chain of operations."""
        def f(x, y, z):
            # stop x, then multiply by y, then multiply by z
            x_stopped = jax.lax.stop_gradient(x)
            return (x_stopped * y) * z

        x = jnp.array([1.0, 2.0])
        y = jnp.array([3.0, 4.0])
        z = jnp.array([5.0, 6.0])

        jac_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
        veres = jac_fn(x, y, z)

        jax_jac_rev = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))
        revres = jax_jac_rev(x, y, z)

        self.assertTrue(tree_allclose(veres, revres))

        # Jacobian wrt x should be 0
        self.assertTrue(jnp.all(veres[0] == 0.0))
        # Jacobian wrt y is diag(x * z)
        self.assertTrue(jnp.all(jnp.diag(veres[1]) == x * z))
        # Jacobian wrt z is diag(x * y)
        self.assertTrue(jnp.all(jnp.diag(veres[2]) == x * y))


if __name__ == "__main__":
    unittest.main()
