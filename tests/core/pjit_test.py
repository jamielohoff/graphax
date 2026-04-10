import unittest

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve, tree_allclose, set_pjit_elimination_order


class PjitTest(unittest.TestCase):
    """Tests for differentiating through jax.jit (pjit primitive)."""

    # ------------------------------------------------------------------
    # Single-argument cases
    # ------------------------------------------------------------------

    def test_single_arg_scalar_fn_fwd(self):
        """Jacobian of a simple scalar-valued jitted function, fwd order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(0)
        x = jrand.normal(key, (4,))

        deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacfwd(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_single_arg_scalar_fn_rev(self):
        """Jacobian of a simple scalar-valued jitted function, rev order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(1)
        x = jrand.normal(key, (4,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_single_arg_matmul_jit(self):
        """Jacobian through a jitted matrix-vector multiply."""
        key = jrand.PRNGKey(2)
        wkey, xkey = jrand.split(key, 2)
        W = jrand.normal(wkey, (3, 4))

        @jax.jit
        def inner(x):
            return jnp.tanh(W @ x)

        def f(x):
            return inner(x)

        x = jrand.normal(xkey, (4,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_single_arg_nested_jit(self):
        """Jacobian through two levels of jax.jit nesting."""
        @jax.jit
        def outer(x):
            @jax.jit
            def inner(x):
                return jnp.exp(x)
            return jnp.cos(inner(x))

        def f(x):
            return outer(x)

        key = jrand.PRNGKey(3)
        x = jrand.normal(key, (3,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_single_arg_jit_inside_larger_graph(self):
        """jitted sub-function appears mid-graph; rest is plain JAX."""
        @jax.jit
        def jit_part(x):
            return jnp.sin(x) * jnp.cos(x)

        def f(x):
            z = jnp.exp(x)
            w = jit_part(z)
            return jnp.tanh(w)

        key = jrand.PRNGKey(4)
        x = jrand.normal(key, (5,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_single_arg_2d_input(self):
        """Jacobian of jitted function with 2-D input."""
        @jax.jit
        def inner(x):
            return jnp.sum(x ** 2, axis=1)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(5)
        x = jrand.normal(key, (3, 4))

        deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacfwd(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # set_pjit_elimination_order round-trip
    # ------------------------------------------------------------------

    def test_set_order_forward(self):
        """Results are the same when the inner jaxpr uses forward elimination."""
        set_pjit_elimination_order("forward")

        @jax.jit
        def inner(x):
            return jnp.log(jnp.abs(x) + 1.)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(6)
        x = jrand.normal(key, (4,)) + 2.

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacrev(f, argnums=(0,))(x)

        set_pjit_elimination_order()  # restore default
        self.assertTrue(tree_allclose(veres, refres))

    def test_set_order_reverse(self):
        """Explicit reverse inner order also matches reference Jacobian."""
        set_pjit_elimination_order("reverse")

        @jax.jit
        def inner(x):
            return jnp.log(jnp.abs(x) + 1.)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(7)
        x = jrand.normal(key, (4,)) + 2.

        deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0,)))
        veres = deriv_fn(x)

        refres = jax.jacfwd(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))


if __name__ == "__main__":
    unittest.main()
