import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.examples import Simple, Helmholtz, f, g


class GeneralADTest(unittest.TestCase): 
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

        print(veres)
        print(revres)
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
            return jnp.sin(x * y)

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
            return x.T + y

        x = jnp.ones((2, 3))
        y = jnp.ones((3, 2))
        jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
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
        
    def test_simple(self):
        x = jnp.ones((50, 50))
        y = jnp.ones((50, 50))

        jacrev_f = jax.jit(jacve(Simple, order="rev", argnums=(0, 1), count_ops=True))
        veres = jacrev_f(x, y)

        jac_f = jax.jit(jax.jacrev(Simple, argnums=(0, 1)))
        revres = jac_f(x, y)

        self.assertTrue(tree_allclose(veres, revres))

    def test_Helmholtz(self):
        x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(4)/2000. # 

        jac_cc = jax.jit(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))
        veres = jac_cc(x)

        jax_jac_fwd = jax.jit(jax.jacfwd(Helmholtz))
        jax_jac_rev = jax.jit(jax.jacfwd(Helmholtz))
        revres = jax_jac_rev(x)
        self.assertTrue(tree_allclose(veres, revres))

    def test_NeuralNetwork(self):
        def NeuralNetwork(x, W1, b1, W2, b2, y):
            y1 = W1 @ x
            z1 = y1 + b1
            a1 = jnp.tanh(z1)
            
            y2 = W2 @ a1
            z2 = y2 + b2
            a2 = jnp.tanh(z2)
            d = a2 - y
            return .5*jnp.sum(d**2)

        key = jrand.PRNGKey(42)

        x = jnp.ones(4)
        y = jrand.normal(key, (4,))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))

        jac_rev = jax.jit(jacve(NeuralNetwork, order="rev", argnums=(1, 2, 3, 4)))
        veres = jac_rev(x, W1, b1, W2, b2, y)

        jax_jac_rev = jax.jit(jax.jacrev(NeuralNetwork, argnums=(1, 2, 3, 4)))
        revres = jax_jac_rev(x, W1, b1, W2, b2, y)

        self.assertTrue(tree_allclose(veres, revres))
    
    def test_vmap_NeuralNetwork(self):
        batchsize = 16
        @partial(jax.vmap, in_axes=(0, None, None, None, None, 0))
        def NeuralNetwork(x, W1, b1, W2, b2, y):
            y1 = W1 @ x
            z1 = y1 + b1
            a1 = jnp.tanh(z1)
            
            y2 = W2 @ a1
            z2 = y2 + b2
            return 0.5*(jnp.tanh(z2) - y)**2
        
        def f(x, W1, b1, W2, b2, y):
            out = NeuralNetwork(x, W1, b1, W2, b2, y)
            return out.sum()
            
        key = jrand.PRNGKey(42)

        x = jnp.ones((batchsize, 4))
        y = jrand.normal(key, (batchsize, 4))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))
        
        print(jax.make_jaxpr(f)(x, W1, b1, W2, b2, y))

        jac_rev = jax.jit(jacve(f, order="rev", argnums=(1, 2, 3, 4)))
        veres = jac_rev(x, W1, b1, W2, b2, y)

        jax_jac_rev = jax.jit(jax.jacrev(f, argnums=(1, 2, 3, 4)))
        revres = jax_jac_rev(x, W1, b1, W2, b2, y)

        self.assertTrue(tree_allclose(veres, revres))
        
    def test_f(self):
        a = jnp.ones(4)
        b = jnp.ones((2, 3))
        c = jnp.ones((4, 4))
        d = jnp.ones((3, 3))
        e = jnp.ones((4, 1))
        xs = [a, b, c, d, e]
        
        print(jax.make_jaxpr(f)(*xs))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2, 3, 4)))
        veres = deriv_fn(*xs)

        revres = jax.jacrev(f, argnums=(0, 1, 2, 3, 4))(*xs)
        
        for i in range(4):
            print("err1", jnp.abs(veres[i][0] - revres[i][0]).mean())
            print("err2", jnp.abs(veres[i][1] - revres[i][1]).mean())
            print("err3", jnp.abs(veres[i][2] - revres[i][2]).mean())
            print("err4", jnp.abs(veres[i][3] - revres[i][3]).mean())

        self.assertTrue(tree_allclose(veres, revres))     
    
    def test_slicing(self):
        def f(x, y):
            z = x @ y
            return jnp.sin(z[:, 0:1])

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 4))
        
        print(jax.make_jaxpr(f)(x, y))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(f, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_squeezing(self):
        def f(x, y):
            z = x @ y
            return z[:, 0].sum()

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 4))
        
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
        
        print(jax.make_jaxpr(f)(x, y, z))

        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
        veres = deriv_fn(x, y, z)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

        self.assertTrue(tree_allclose(veres, revres)) 
        
    def test_concatenate_2(self):
        def f(x, y, z):
            w = jnp.concatenate([x, y, z], axis=0)
            return jnp.sin(w)

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        y = jrand.normal(ykey, (2,))
        z = jrand.normal(ykey, (3,))
        
        print(jax.make_jaxpr(f)(x, y, z))

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
        
        print(jax.make_jaxpr(f)(x, y))
        print(jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y))
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

if __name__ == '__main__':
    unittest.main()

