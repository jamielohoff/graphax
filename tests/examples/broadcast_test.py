"""
Tests for duplicate broadcast_in_dim bug in vertex elimination.

When the same 1D variable is broadcast to 2D via two separate broadcast_in_dim
operations and used in separate sub operations that are later combined, the
SparseTensor shapes must be compatible for addition/multiplication.

Failing jaxpr pattern:
    b = reduce_sum(a)          # shape [4]
    c = broadcast_in_dim(b)    # shape [1,4]  (first broadcast)
    d = sub(a, c)              # shape [4,4]
    e = broadcast_in_dim(b)    # shape [1,4]  (second broadcast of SAME variable)
    f = sub(a, e)              # shape [4,4]
    g = add(d, f)              # combine

This is reproduced in layer norm where (x - mu) appears twice in the expression.
"""
import unittest
from functools import partial
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve, tree_allclose


def test_order(order: str | Sequence[int], fn: Callable, argnums: Sequence[int],
               *args) -> bool:
    jacve_f = jax.jit(jacve(fn, order=order, argnums=argnums, count_ops=True))
    veres, aux = jacve_f(*args)
            
    jacrev_f = jax.jit(jax.jacrev(fn, argnums=argnums))
    revres = jacrev_f(*args)

    return tree_allclose(veres, revres)

test_rev = partial(test_order, "rev")

def test_fwd(fn: Callable, argnums: Sequence[int], *args) -> bool:
    jacve_f = jax.jit(jacve(fn, order="fwd", argnums=argnums, count_ops=True))
    veres, aux = jacve_f(*args)

    jacfwd_f = jax.jit(jax.jacfwd(fn, argnums=argnums))
    fwdres = jacfwd_f(*args)

    print(fwdres)
    print('#'*80)
    print(veres)

    return tree_allclose(veres, fwdres)


class BroadcastBugTests(unittest.TestCase):
    """Tests for the duplicate broadcast_in_dim vertex elimination bug."""

    # def test_duplicate_broadcast_rev(self):
    #     """Test that duplicate broadcast_in_dim vertices are handled correctly in reverse mode.
        
    #     When the same 1D variable is broadcast to 2D via two separate broadcast_in_dim
    #     operations and used in separate sub operations that are later combined, the
    #     SparseTensor shapes must be compatible for addition.
    #     """
    #     print("Testing duplicate_broadcast_rev...")
    #     def fn(x):
    #         s = jnp.sum(x, axis=-1)
    #         a = x - s
    #         b = x - s
    #         return a + b
    #     x = jnp.ones((4, 4))
    #     self.assertTrue(test_rev(fn, (0,), x))

    # def test_duplicate_broadcast_fwd(self):
    #     """Same as test_duplicate_broadcast_rev but in forward mode."""
    #     print("Testing duplicate_broadcast_fwd...")
    #     def fn(x):
    #         s = jnp.sum(x, axis=-1)
    #         a = x - s
    #         b = x - s
    #         return a + b
    #     x = jnp.ones((4, 4))
    #     self.assertTrue(test_fwd(fn, (0,), x))

    # def test_duplicate_broadcast_div_rev(self):
    #     """Test duplicate broadcast with division in reverse mode."""
    #     print("Testing duplicate_broadcast_div_rev...")
    #     def fn(x):
    #         s = jnp.sum(x, axis=-1)
    #         a = x / s
    #         b = x / s
    #         return a + b
    #     x = jnp.ones((4, 4))
    #     self.assertTrue(test_rev(fn, (0,), x))

    # def test_duplicate_broadcast_mul_rev(self):
    #     """Test duplicate broadcast with multiplication in reverse mode."""
    #     print("Testing duplicate_broadcast_mul_rev...")
    #     def fn(x):
    #         s = jnp.sum(x, axis=-1)
    #         a = x * s
    #         b = x * s
    #         return a + b
    #     x = jnp.ones((4, 4))
    #     self.assertTrue(test_rev(fn, (0,), x))

    # def test_layer_norm_rev(self):
    #     """Test layer norm with duplicate (x - mu) usage in reverse mode.
        
    #     Layer norm computes (x - mu) twice: once for the numerator and once
    #     for the variance, creating two separate broadcast_in_dim vertices.
    #     """
    #     print("Testing layer_norm_rev...")
    #     def fn(x, gamma, beta):
    #         mu = jnp.mean(x, axis=-1)
    #         sigma = jnp.sum((x - mu)**2, axis=-1)/x.shape[-1]
    #         normed = (x - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta
    #         return normed
    #     x = jnp.ones((4, 4))
    #     gamma = jnp.ones(4)
    #     beta = jnp.zeros(4)
    #     self.assertTrue(test_rev(fn, (0, 1, 2), x, gamma, beta))

    # def test_layer_norm_fwd(self):
    #     """Test layer norm with duplicate (x - mu) usage in forward mode."""
    #     print("Testing layer_norm_fwd...")
    #     def fn(x, gamma, beta):
    #         mu = jnp.mean(x, axis=-1)
    #         sigma = jnp.sum((x - mu)**2, axis=-1)/x.shape[-1]
    #         normed = (x - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta
    #         return normed
    #     x = jnp.ones((4, 4))
    #     gamma = jnp.ones(4)
    #     beta = jnp.zeros(4)
    #     self.assertTrue(test_fwd(fn, (0, 1, 2), x, gamma, beta))

    def test_two_blocks(self):
        """Test two encoder blocks in forward mode."""
        print("Testing two_blocks_fwd...")
        def encoder_block(x, WQ, WK, WV, W, b, gamma, beta):
            q = x @ WQ
            k = x @ WK
            v = x @ WV
            a = q @ k.T
            z = jax.nn.softmax(a, axis=1) / jnp.sqrt(k.shape[-1])
            att = z @ v
            r = x + att
            mu = jnp.mean(r, axis=-1, keepdims=True)
            sigma = jnp.var((r - mu)**2, axis=-1, keepdims=True) / r.shape[-1]
            normed = (r - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta
            return jnp.tanh(normed @ W + b)
        def two_blocks(x, WQ1, WK1, WV1, W1, b1, gamma0, beta0, WQ2, WK2, WV2, W2, b2, gamma1, beta1):
            z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, gamma0, beta0)
            z2 = encoder_block(z1, WQ2, WK2, WV2, W2, b2, gamma1, beta1)
            return jnp.sum(z2)
        key = jrand.PRNGKey(42)
        x = jnp.ones((5, 4))
        WQ1 = jrand.normal(key, (4, 4))
        WK1 = jrand.normal(key, (4, 4))
        WV1 = jrand.normal(key, (4, 4))
        W1 = jrand.normal(key, (4, 4))
        b1 = jrand.normal(key, (4,))
        WQ2 = jrand.normal(key, (4, 4))
        WK2 = jrand.normal(key, (4, 4))
        WV2 = jrand.normal(key, (4, 4))
        W2 = jrand.normal(key, (4, 4))
        b2 = jrand.normal(key, (4,))
        gamma0 = jnp.ones(4)
        beta0 = jnp.zeros(4)
        gamma1 = jnp.ones(4)
        beta1 = jnp.zeros(4)
        args = (x, WQ1, WK1, WV1, W1, b1, gamma0, beta0, WQ2, WK2, WV2, W2, b2, gamma1, beta1)
        argnums = list(range(15))
        two_blocks(*args)
        self.assertTrue(test_fwd(two_blocks, argnums, *args))
        self.assertTrue(test_rev(two_blocks, argnums, *args))


    # def test_attn_layernorm_rev(self):
    #     """Test attention + layer norm pattern in reverse mode.
        
    #     This reproduces the exact error from the Encoder test.
    #     """
    #     print("Testing attn_layernorm_rev...")
    #     def attn_layernorm(x, WQ, WK, WV, gamma, beta):
    #         q = WQ @ x
    #         k = WK @ x
    #         v = WV @ x
    #         a = q.T @ k
    #         z = jax.nn.softmax(a, axis=1)
    #         att = z @ v
    #         r = x + att
    #         mu = jnp.mean(r, axis=-1)
    #         sigma = jnp.sum((r - mu)**2, axis=-1)/r.shape[-1]
    #         normed = (r - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta
    #         return normed
    #     key = jrand.PRNGKey(42)
    #     x = jnp.ones((4, 4))
    #     WQ = jrand.normal(jrand.PRNGKey(1), (4, 4))
    #     WK = jrand.normal(jrand.PRNGKey(2), (4, 4))
    #     WV = jrand.normal(jrand.PRNGKey(3), (4, 4))
    #     gamma = jnp.ones(4)
    #     beta = jnp.zeros(4)
    #     args = (x, WQ, WK, WV, gamma, beta)
    #     argnums = (0, 1, 2, 3, 4, 5)
    #     self.assertTrue(test_rev(attn_layernorm, argnums, *args))


if __name__ == "__main__":
    unittest.main()
