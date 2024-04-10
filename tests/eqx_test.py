import unittest

import jax
import jax.numpy as jnp

import equinox as eqx
from graphax import filter_jacve, tree_allclose


class TestEqx(unittest.TestCase):
    def test_simple_NN_and_loss(self):
        key = jax.random.PRNGKey(42)
        model = eqx.nn.MLP(10, 5, 64, 3, activation=jnp.tanh, key=key)

        def loss_fn(model, x, y):
            return jnp.mean((jax.vmap(model)(x) - y) ** 2)

        x = jnp.ones((64, 10))
        y = jnp.ones((64, 5), dtype=jnp.int32)

        jacve_res = filter_jacve(loss_fn, order="rev")(model, x, y)
        jax_res_fn = eqx.filter_grad(loss_fn)
        jax_res = jax_res_fn(model, x, y)
        self.assertTrue(tree_allclose(jacve_res, jax_res))

    def test_simple_NN(self):
        key = jax.random.PRNGKey(42)
        model = eqx.nn.MLP(10, 5, 64, 3, activation=jnp.tanh, key=key)

        def loss_fn(model, x):
            return model(x)

        x = jnp.ones((64, 10))

        jacve_res = filter_jacve(loss_fn, order="rev")(model, x)
        jax_res_fn = eqx.filter_grad(loss_fn)
        jax_res = jax_res_fn(model, x)
        
        self.assertTrue(tree_allclose(jacve_res, jax_res))

