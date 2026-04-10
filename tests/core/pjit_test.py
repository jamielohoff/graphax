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


class PjitMultiOutputTest(unittest.TestCase):
    """Tests for differentiating through jax.jit with multiple outputs."""

    # ------------------------------------------------------------------
    # Basic 2-output cases
    # ------------------------------------------------------------------

    def test_two_returns_fwd(self):
        """jit returning a 2-tuple; both outputs differentiated, fwd order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x), jnp.cos(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(10)
        x = jrand.normal(key, (4,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_two_returns_rev(self):
        """jit returning a 2-tuple; both outputs differentiated, rev order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x), jnp.cos(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(11)
        x = jrand.normal(key, (4,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # Multiple inputs × multiple outputs
    # ------------------------------------------------------------------

    def test_two_returns_two_inputs_fwd(self):
        """jit with 2 inputs returning 2 outputs, fwd order."""
        @jax.jit
        def inner(x, y):
            return jnp.sin(x) + y, jnp.cos(x) * y

        def f(x, y):
            return inner(x, y)

        key = jrand.PRNGKey(12)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3,))
        y = jrand.normal(ykey, (3,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))(x, y)
        refres = jax.jacfwd(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, refres))

    def test_two_returns_two_inputs_rev(self):
        """jit with 2 inputs returning 2 outputs, rev order."""
        @jax.jit
        def inner(x, y):
            return jnp.sin(x) + y, jnp.cos(x) * y

        def f(x, y):
            return inner(x, y)

        key = jrand.PRNGKey(13)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3,))
        y = jrand.normal(ykey, (3,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0, 1)))(x, y)
        refres = jax.jacrev(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # Only one jit output consumed downstream
    # ------------------------------------------------------------------

    def test_one_output_used_as_intermediate_fwd(self):
        """jit returns 2 values; only the first is used in subsequent ops."""
        @jax.jit
        def inner(x):
            return jnp.exp(x), jnp.tanh(x)

        def f(x):
            a, _b = inner(x)
            return jnp.sum(a)

        key = jrand.PRNGKey(14)
        x = jrand.normal(key, (5,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_one_output_used_as_intermediate_rev(self):
        """jit returns 2 values; only the second is used in subsequent ops."""
        @jax.jit
        def inner(x):
            return jnp.exp(x), jnp.tanh(x)

        def f(x):
            _a, b = inner(x)
            return jnp.sum(b)

        key = jrand.PRNGKey(15)
        x = jrand.normal(key, (5,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # 3 outputs
    # ------------------------------------------------------------------

    def test_three_returns_fwd(self):
        """jit returning a 3-tuple, fwd order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x), jnp.cos(x), jnp.tanh(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(16)
        x = jrand.normal(key, (4,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_three_returns_rev(self):
        """jit returning a 3-tuple, rev order."""
        @jax.jit
        def inner(x):
            return jnp.sin(x), jnp.cos(x), jnp.tanh(x)

        def f(x):
            return inner(x)

        key = jrand.PRNGKey(17)
        x = jrand.normal(key, (4,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # Multi-output jit embedded inside a larger computation graph
    # ------------------------------------------------------------------

    def test_multi_output_in_larger_graph_fwd(self):
        """Multi-output jit; each output is used in further non-jitted ops."""
        @jax.jit
        def jit_part(x):
            return jnp.sin(x), jnp.exp(x)

        def f(x):
            z = jnp.log(jnp.abs(x) + 1.)
            a, b = jit_part(z)
            return jnp.tanh(a) + jnp.cos(b)

        key = jrand.PRNGKey(18)
        x = jrand.normal(key, (4,)) + 2.

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_multi_output_in_larger_graph_rev(self):
        """Multi-output jit mid-graph, rev order."""
        @jax.jit
        def jit_part(x):
            return jnp.sin(x), jnp.exp(x)

        def f(x):
            z = jnp.log(jnp.abs(x) + 1.)
            a, b = jit_part(z)
            return jnp.tanh(a) + jnp.cos(b)

        key = jrand.PRNGKey(19)
        x = jrand.normal(key, (4,)) + 2.

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f, argnums=(0,))(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # Nested jit with multiple outputs
    # ------------------------------------------------------------------

    def test_nested_jit_multi_output_fwd(self):
        """Outer jit with multiple outputs, each computed by an inner jit."""
        @jax.jit
        def outer(x):
            @jax.jit
            def inner_sin(x):
                return jnp.sin(x)
            @jax.jit
            def inner_cos(x):
                return jnp.cos(x)
            return inner_sin(x), inner_cos(x)

        def f(x):
            return outer(x)

        key = jrand.PRNGKey(20)
        x = jrand.normal(key, (3,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_nested_jit_multi_output_rev(self):
        """Outer jit with multiple outputs, nested jits inside, rev order."""
        @jax.jit
        def outer(x):
            @jax.jit
            def inner_sin(x):
                return jnp.sin(x)
            @jax.jit
            def inner_cos(x):
                return jnp.cos(x)
            return inner_sin(x), inner_cos(x)

        def f(x):
            return outer(x)

        key = jrand.PRNGKey(21)
        x = jrand.normal(key, (3,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    # ------------------------------------------------------------------
    # 2-D inputs
    # ------------------------------------------------------------------

    def test_2d_input_multi_output_fwd(self):
        """Multi-output jit with 2-D inputs, fwd order."""
        key = jrand.PRNGKey(22)
        wkey, xkey = jrand.split(key, 2)
        W = jrand.normal(wkey, (3, 4))

        @jax.jit
        def inner(x):
            return jnp.tanh(W @ x), jnp.sum(x ** 2, keepdims=True)

        def f(x):
            return inner(x)

        x = jrand.normal(xkey, (4,))

        veres = jax.jit(jacve(f, order="fwd", argnums=(0,)))(x)
        refres = jax.jacfwd(f)(x)

        self.assertTrue(tree_allclose(veres, refres))

    def test_2d_input_multi_output_rev(self):
        """Multi-output jit with 2-D weight matrix, rev order."""
        key = jrand.PRNGKey(23)
        wkey, xkey = jrand.split(key, 2)
        W = jrand.normal(wkey, (3, 4))

        @jax.jit
        def inner(x):
            return jnp.tanh(W @ x), jnp.sum(x ** 2, keepdims=True)

        def f(x):
            return inner(x)

        x = jrand.normal(xkey, (4,))

        veres = jax.jit(jacve(f, order="rev", argnums=(0,)))(x)
        refres = jax.jacrev(f)(x)

        self.assertTrue(tree_allclose(veres, refres))


if __name__ == "__main__":
    unittest.main()
