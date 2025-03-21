import unittest

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose


class PrimitveTest(unittest.TestCase): 
    def test_broadcast_add(self):
        def broadcast_add(x, y):
            return jnp.tanh(x + y)

        x = 2*jnp.ones((2, 3))
        y = 3*jnp.ones((1, 3))
        jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        self.assertTrue(tree_allclose(veres, revres))
    
    def test_broadcast_sub(self):
        def broadcast_add(x, y):
            return jnp.tanh(x - y)

        x = 2*jnp.ones((2, 3))
        y = 3*jnp.ones((1, 3))
        jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        self.assertTrue(tree_allclose(veres, revres))
        
    def test_broadcast_mul(self):
        def broadcast_mul(x, y):
            z = jnp.exp(y)
            return jnp.sin(x * z)

        x = jnp.arange(6).reshape((2, 3)).astype(jnp.float32)
        y = jnp.arange(3).reshape((3, )).astype(jnp.float32)
        jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)
        get_shape = lambda x: x.shape

        self.assertTrue(tree_allclose(veres, revres))
    
    def test_broadcast_outer_product(self):
        def broadcast_mul(x, y):
            return jnp.sin(x * y)

        x = jnp.arange(4).reshape((4, 1)).astype(jnp.float32) + 1
        y = jnp.arange(3).reshape((1, 3)).astype(jnp.float32)
        jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)
        get_shape = lambda x: x.shape

        self.assertTrue(tree_allclose(veres, revres))

    def test_transpose(self):
        def transpose(x, y):
            z = jnp.cos(x)
            return z.T * y

        x = jnp.ones((2, 3))
        y = jnp.ones((3, 2))
        jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
        jaxpr = jax.make_jaxpr(jac_fwd)(x, y)
        veres = jac_fwd(x, y)[0]

        revres = jax.jacrev(transpose)(x, y)

        self.assertTrue(tree_allclose(veres, revres))
    
    def test_matmul(self):
        def f(x, y):
            z = x @ y
            return jnp.sin(z)

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres))
    
    def test_reduce_sum(self):
        def sums(x, y):
            return jnp.sin(jnp.sum(x@y, axis=0))

        x = jnp.ones((2, 3))
        y = jnp.ones((3, 4))
        
        jac_fwd = jax.jit(jacve(sums, order="rev", argnums=(0, 1)))
        veres = jac_fwd(x, y)

        revres = jax.jacrev(sums, argnums=(0, 1))(x, y)
    
        self.assertTrue(tree_allclose(veres, revres))
        
    def test_reduce_max(self):
        def maxs(x, y):
            return jnp.sin(jnp.max(x@y, axis=0))

        x = jnp.array([[0., 1., 2.],[1., 0., 2.]])
        y = jnp.ones((3, 4))
        
        jac_rev = jax.jit(jacve(maxs, order="rev", argnums=(0, 1)))
        veres = jac_rev(x, y)

        revres = jax.jacrev(maxs, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres))
        
    def test_slicing(self):
        def f(x, y):
            z = x @ y
            return jnp.sin(z[:, 0:1])

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 4))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_squeezing(self):
        def f(x, y):
            z = x @ y
            return jnp.squeeze(z).sum()

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 1))
        
        print(jax.make_jaxpr(f)(x, y))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_concatenate_1(self):
        def f(x, y, z):
            z = jnp.concatenate([y, z], axis=0)
            w = x @ z
            return jnp.sin(w)

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (2, 4))
        z = jrand.normal(ykey, (1, 4))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
        veres = deriv_fn(x, y, z)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_concatenate_2(self):
        def f(x, y, z):
            x = jnp.sin(x)
            y = jnp.cos(y)
            z = jnp.tanh(z)
            w = jnp.concatenate([x, y, z], axis=0)
            return jnp.sin(w)

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        y = jrand.normal(ykey, (2,))
        z = jrand.normal(ykey, (3,))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
        veres = deriv_fn(x, y, z)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_reshape(self):
        def f(x, y):
            x = jnp.reshape(x, (2, 3))
            return jnp.sin(x @ y)

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (6,))
        y = jrand.normal(ykey, (3,))
        
        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)

        self.assertTrue(tree_allclose(veres, revres)) 
    
    def test_large_matmul(self):
        def f(x, y):
            return lax.dot_general(x, y, (([2], [0]), ([0], [1])))

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 1, 4))
        y = jrand.normal(ykey, (4, 3, 2))
        
        print("result", f(x, y).shape)
        print(jax.make_jaxpr(f)(x, y))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
        
        print("err1", jnp.abs(veres[0] - revres[0]).mean())
        print("err2", jnp.abs(veres[1] - revres[1]).mean())
        
        self.assertTrue(tree_allclose(veres, revres))   
        
    def test_eq(self):
        def f(x, y):
            w = jnp.sin(x)
            z = jnp.sin(y)
            return (w == z) - 1.
        x = jnp.array([[1., 0., 1.]])
        y = jnp.array([[1.], [0.], [0.]])

        deriv_fn = jacve(f, order="rev", argnums=(0, 1))
        veres = deriv_fn(x, y)
        
        jax_deriv_fn = jax.jacrev(f, argnums=(0, 1))
        revres = jax_deriv_fn(x, y)
        
        self.assertTrue(tree_allclose(veres, revres)) 
        
        