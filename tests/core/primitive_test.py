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
        print(jax.make_jaxpr(broadcast_add)(x, y))
        print(jax.make_jaxpr(jacve(broadcast_add, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(broadcast_add, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        print(veres[0].shape)
        print(revres[0].shape)
        self.assertTrue(tree_allclose(veres, revres))
    
    def test_broadcast_sub(self):
        def broadcast_add(x, y):
            return jnp.tanh(x - y)

        x = 2*jnp.ones((2, 3))
        y = 3*jnp.ones((1, 3))
        print(jax.make_jaxpr(broadcast_add)(x, y))
        print(jax.make_jaxpr(jacve(broadcast_add, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(broadcast_add, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)

        print(veres)
        print(revres)
        self.assertTrue(tree_allclose(veres, revres))
        
    def test_broadcast_mul(self):
        def broadcast_mul(x, y):
            z = jnp.exp(y)
            return jnp.sin(x * z)

        x = jnp.arange(6).reshape((2, 3)).astype(jnp.float32)
        y = jnp.arange(3).reshape((3, )).astype(jnp.float32)
        print(jax.make_jaxpr(broadcast_mul)(x, y))
        print(jax.make_jaxpr(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(broadcast_mul, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)
        get_shape = lambda x: x.shape

        print(veres[1])
        print(revres[0])
        
        print(tree_map(get_shape, veres), tree_map(get_shape, revres))
        self.assertTrue(tree_allclose(veres, revres))
    
    def test_broadcast_outer_product(self):
        def broadcast_mul(x, y):
            return jnp.sin(x * y)

        x = jnp.arange(4).reshape((4, 1)).astype(jnp.float32) + 1
        y = jnp.arange(3).reshape((1, 3)).astype(jnp.float32)
        print(jax.make_jaxpr(broadcast_mul)(x, y))
        print(jax.make_jaxpr(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))(x, y))
        jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
        veres = jac_rev(x, y)

        print(jax.make_jaxpr(jax.jacfwd(broadcast_mul, argnums=(0, 1)))(x, y))
        jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
        revres = jax_jac_rev(x, y)
        get_shape = lambda x: x.shape

        print(veres[0])
        print(revres[0])
        
        print(tree_map(get_shape, veres), tree_map(get_shape, revres))
        self.assertTrue(tree_allclose(veres, revres))

    def test_transpose(self):
        def transpose(x, y):
            z = jnp.cos(x)
            return z.T * y

        x = jnp.ones((2, 3))
        y = jnp.ones((3, 2))
        jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
        jaxpr = jax.make_jaxpr(jac_fwd)(x, y)
        print(jaxpr)
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
        
        print(jax.make_jaxpr(sums)(x, y))
        
        jac_fwd = jax.jit(jacve(sums, order="rev", argnums=(0, 1)))
        veres = jac_fwd(x, y)

        revres = jax.jacrev(sums, argnums=(0, 1))(x, y)
        
        print(veres, revres)

        self.assertTrue(tree_allclose(veres, revres))
        
    def test_reduce_max(self):
        def maxs(x, y):
            return jnp.sin(jnp.max(x@y, axis=0))

        x = jnp.array([[0., 1., 2.],[1., 0., 2.]])
        y = jnp.ones((3, 4))
        
        print(jax.make_jaxpr(maxs)(x, y))
        
        jac_rev = jax.jit(jacve(maxs, order="rev", argnums=(0, 1)))
        veres = jac_rev(x, y)

        revres = jax.jacrev(maxs, argnums=(0, 1))(x, y)
        
        print(veres)
        print(revres[0])
        print(revres[0].shape)

        self.assertTrue(tree_allclose(veres, revres))
        
        