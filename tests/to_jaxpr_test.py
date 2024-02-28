import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.examples import (Simple, Helmholtz, f, g, RoeFlux_1d, RobotArm_6DOF,
                              EncoderDecoder, Lighthouse)
from graphax.sparse.utils import count_muls, count_muls_jaxpr


class GeneralADTest(unittest.TestCase): 
    # def test_broadcast_add(self):
    #     def broadcast_add(x, y):
    #         return jnp.tanh(x + y)

    #     x = 2*jnp.ones((2, 3))
    #     y = 3*jnp.ones((1, 3))
    #     print(jax.make_jaxpr(broadcast_add)(x, y))
    #     print(jax.make_jaxpr(jacve(broadcast_add, order="fwd", argnums=(0, 1)))(x, y))
    #     jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     print(jax.make_jaxpr(jax.jacfwd(broadcast_add, argnums=(0, 1)))(x, y))
    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)

    #     print(veres)
    #     print(revres)
    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_broadcast_sub(self):
    #     def broadcast_add(x, y):
    #         return jnp.tanh(x - y)

    #     x = 2*jnp.ones((2, 3))
    #     y = 3*jnp.ones((1, 3))
    #     print(jax.make_jaxpr(broadcast_add)(x, y))
    #     print(jax.make_jaxpr(jacve(broadcast_add, order="fwd", argnums=(0, 1)))(x, y))
    #     jac_rev = jax.jit(jacve(broadcast_add, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     print(jax.make_jaxpr(jax.jacfwd(broadcast_add, argnums=(0, 1)))(x, y))
    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_add, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)

    #     print(veres)
    #     print(revres)
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_broadcast_mul(self):
    #     def broadcast_mul(x, y):
    #         return jnp.sin(x * y)

    #     x = jnp.arange(6).reshape((2, 3)).astype(jnp.float32)
    #     y = jnp.arange(3).reshape((3, )).astype(jnp.float32)
    #     print(jax.make_jaxpr(broadcast_mul)(x, y))
    #     print(jax.make_jaxpr(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))(x, y))
    #     jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     print(jax.make_jaxpr(jax.jacfwd(broadcast_mul, argnums=(0, 1)))(x, y))
    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)
    #     get_shape = lambda x: x.shape

    #     print(veres[1])
    #     print(revres[0])
        
    #     print(tree_map(get_shape, veres), tree_map(get_shape, revres))
    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_broadcast_outer_product(self):
    #     def broadcast_mul(x, y):
    #         return jnp.sin(x * y)

    #     x = jnp.arange(4).reshape((4, 1)).astype(jnp.float32) + 1
    #     y = jnp.arange(3).reshape((1, 3)).astype(jnp.float32)
    #     print(jax.make_jaxpr(broadcast_mul)(x, y))
    #     print(jax.make_jaxpr(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))(x, y))
    #     jac_rev = jax.jit(jacve(broadcast_mul, order="fwd", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     print(jax.make_jaxpr(jax.jacfwd(broadcast_mul, argnums=(0, 1)))(x, y))
    #     jax_jac_rev = jax.jit(jax.jacfwd(broadcast_mul, argnums=(0, 1)))
    #     revres = jax_jac_rev(x, y)
    #     get_shape = lambda x: x.shape

    #     print(veres[0])
    #     print(revres[0])
        
    #     print(tree_map(get_shape, veres), tree_map(get_shape, revres))
    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_transpose(self):
    #     def transpose(x, y):
    #         return x.T + y

    #     x = jnp.ones((2, 3))
    #     y = jnp.ones((3, 2))
    #     jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
    #     veres = jac_fwd(x, y)[0]

    #     revres = jax.jacrev(transpose)(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_matmul(self):
    #     def f(x, y):
    #         z = x @ y
    #         return jnp.sin(z)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3,))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_reduce_sum(self):
    #     def sums(x, y):
    #         return jnp.sin(jnp.sum(x@y, axis=0))

    #     x = jnp.ones((2, 3))
    #     y = jnp.ones((3, 4))
        
    #     print(jax.make_jaxpr(sums)(x, y))
        
    #     jac_fwd = jax.jit(jacve(sums, order="rev", argnums=(0, 1)))
    #     veres = jac_fwd(x, y)

    #     revres = jax.jacrev(sums, argnums=(0, 1))(x, y)
        
    #     print(veres, revres)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_reduce_max(self):
    #     def maxs(x, y):
    #         return jnp.sin(jnp.max(x@y, axis=0))

    #     x = jnp.array([[0., 1., 2.],[1., 0., 2.]])
    #     y = jnp.ones((3, 4))
        
    #     print(jax.make_jaxpr(maxs)(x, y))
        
    #     jac_rev = jax.jit(jacve(maxs, order="rev", argnums=(0, 1)))
    #     veres = jac_rev(x, y)

    #     revres = jax.jacrev(maxs, argnums=(0, 1))(x, y)
        
    #     print(veres)
    #     print(revres[0])
    #     print(revres[0].shape)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_simple(self):
    #     x = jnp.ones((50, 50))
    #     y = jnp.ones((50, 50))
        
    #     jacrev_f = jax.jit(jacve(Simple, order="rev", argnums=(0, 1)))
    #     veres = jacrev_f(x, y)
                
    #     jac_f = jax.jit(jax.jacrev(Simple, argnums=(0, 1)))
    #     revres = jac_f(x, y)
        
    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_Helmholtz(self):
    #     x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(4)/2000. # 

    #     jac_cc = jax.jit(jacve(Helmholtz, order=[2, 5, 4, 3, 1], count_ops=True))
    #     veres, aux = jac_cc(x)
                
    #     jaxpr = jax.make_jaxpr(Helmholtz)(x)
    #     deriv_jaxpr = jax.make_jaxpr(jacve(Helmholtz, order=[2, 5, 4, 3, 1], argnums=(0,), dense_representation=False))(x)
    #     print(deriv_jaxpr)
    #     print("num_muls", aux["num_muls"])
    #     print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

    #     jax_jac_rev = jax.jit(jax.jacrev(Helmholtz))
    #     revres = jax_jac_rev(x)
    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_NeuralNetwork(self):
    #     def NeuralNetwork(x, W1, b1, W2, b2, y):
    #         y1 = W1 @ x
    #         z1 = y1 + b1
    #         a1 = jnp.tanh(z1)
            
    #         y2 = W2 @ a1
    #         z2 = y2 + b2
    #         a2 = jnp.tanh(z2)
    #         d = a2 - y
    #         return .5*jnp.sum(d**2)

    #     key = jrand.PRNGKey(42)

    #     x = jnp.ones(4)
    #     y = jrand.normal(key, (4,))

    #     w1key, b1key, key = jrand.split(key, 3)
    #     W1 = jrand.normal(w1key, (8, 4))
    #     b1 = jrand.normal(b1key, (8,))

    #     w2key, b2key, key = jrand.split(key, 3)
    #     W2 = jrand.normal(w2key, (4, 8))
    #     b2 = jrand.normal(b2key, (4,))

    #     jac_rev = jax.jit(jacve(NeuralNetwork, order="rev", argnums=(1, 2, 3, 4)))
    #     veres = jac_rev(x, W1, b1, W2, b2, y)

    #     jax_jac_rev = jax.jit(jax.jacrev(NeuralNetwork, argnums=(1, 2, 3, 4)))
    #     revres = jax_jac_rev(x, W1, b1, W2, b2, y)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_vmap_NeuralNetwork(self):
    #     batchsize = 16
    #     @partial(jax.vmap, in_axes=(0, None, None, None, None, 0))
    #     def NeuralNetwork(x, W1, b1, W2, b2, y):
    #         y1 = W1 @ x
    #         z1 = y1 + b1
    #         a1 = jnp.tanh(z1)
            
    #         y2 = W2 @ a1
    #         z2 = y2 + b2
    #         return 0.5*(jnp.tanh(z2) - y)**2
        
    #     def f(x, W1, b1, W2, b2, y):
    #         out = NeuralNetwork(x, W1, b1, W2, b2, y)
    #         return out.sum()
            
    #     key = jrand.PRNGKey(42)

    #     x = jnp.ones((batchsize, 4))
    #     y = jrand.normal(key, (batchsize, 4))

    #     w1key, b1key, key = jrand.split(key, 3)
    #     W1 = jrand.normal(w1key, (8, 4))
    #     b1 = jrand.normal(b1key, (8,))

    #     w2key, b2key, key = jrand.split(key, 3)
    #     W2 = jrand.normal(w2key, (4, 8))
    #     b2 = jrand.normal(b2key, (4,))
        
    #     print(jax.make_jaxpr(f)(x, W1, b1, W2, b2, y))

    #     jac_rev = jax.jit(jacve(f, order="rev", argnums=(1, 2, 3, 4)))
    #     veres = jac_rev(x, W1, b1, W2, b2, y)

    #     jax_jac_rev = jax.jit(jax.jacrev(f, argnums=(1, 2, 3, 4)))
    #     revres = jax_jac_rev(x, W1, b1, W2, b2, y)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_f(self):
    #     a = jnp.ones(4)
    #     b = jnp.ones((2, 3))
    #     c = jnp.ones((4, 4))
    #     d = jnp.ones((3, 3))
    #     e = jnp.ones((4, 1))
    #     xs = [a, b, c, d, e]
        
    #     print(jax.make_jaxpr(f)(*xs))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2, 3, 4)))
    #     veres = deriv_fn(*xs)

    #     revres = jax.jacrev(f, argnums=(0, 1, 2, 3, 4))(*xs)
        
    #     for i in range(4):
    #         print("err1", jnp.abs(veres[i][0] - revres[i][0]).mean())
    #         print("err2", jnp.abs(veres[i][1] - revres[i][1]).mean())
    #         print("err3", jnp.abs(veres[i][2] - revres[i][2]).mean())
    #         print("err4", jnp.abs(veres[i][3] - revres[i][3]).mean())

    #     self.assertTrue(tree_allclose(veres, revres))     
    
    # def test_slicing(self):
    #     def f(x, y):
    #         z = x @ y
    #         return jnp.sin(z[:, 0:1])

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))
        
    #     print(jax.make_jaxpr(f)(x, y))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_squeezing(self):
    #     def f(x, y):
    #         z = x @ y
    #         return z[:, 0].sum()

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))
        
    #     print(jax.make_jaxpr(f)(x, y))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_concatenate_1(self):
    #     def f(x, y, z):
    #         z = jnp.concatenate([y, z], axis=0)
    #         w = x @ z
    #         return jnp.sin(w)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (2, 4))
    #     z = jrand.normal(ykey, (1, 4))
        
    #     print(jax.make_jaxpr(f)(x, y, z))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
    #     veres = deriv_fn(x, y, z)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

    #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # # def test_concatenate_2(self):
    # #     def f(x, y, z):
    # #         w = jnp.concatenate([x, y, z], axis=0)
    # #         return jnp.sin(w)

    # #     key = jrand.PRNGKey(42)
    # #     xkey, ykey = jrand.split(key, 2)
    # #     x = jrand.normal(xkey, (4,))
    # #     y = jrand.normal(ykey, (2,))
    # #     z = jrand.normal(ykey, (3,))
        
    # #     print(jax.make_jaxpr(f)(x, y, z))

    # #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1, 2)))
    # #     veres = deriv_fn(x, y, z)

    # #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1, 2)))(x, y, z)

    # #     self.assertTrue(tree_allclose(veres, revres)) 
        
    # def test_reshape(self):
    #     def f(x, y):
    #         x = jnp.reshape(x, (2, 3))
    #         return jnp.sin(x @ y)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (6,))
    #     y = jrand.normal(ykey, (3,))
        
    #     print(jax.make_jaxpr(f)(x, y))
    #     print(jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y))
    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)

    #     self.assertTrue(tree_allclose(veres, revres)) 
    
    # def test_large_matmul(self):
    #     def f(x, y):
    #         return lax.dot_general(x, y, (([2], [0]), ([0], [1])))

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (3, 1, 4))
    #     y = jrand.normal(ykey, (4, 3, 2))
        
    #     print("result", f(x, y).shape)
    #     print(jax.make_jaxpr(f)(x, y))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).mean())
    #     print("err2", jnp.abs(veres[1] - revres[1]).mean())
        
    #     self.assertTrue(tree_allclose(veres, revres))    
        
    # def test_RoeFlux1d(self):
    #     xs = [.02]*6
    #     order = [37, 3, 27, 49, 63, 68, 43, 81, 76, 15, 44, 88, 25, 73, 12, 70, 
    #             57, 1, 24, 36, 67, 30, 92, 42, 78, 91, 31, 16, 80, 7, 32, 28, 56, 
    #             8, 39, 79, 58, 33, 47, 83, 38, 90, 87, 74, 85, 26, 94, 4, 13, 17, 
    #             14, 93, 61, 72, 97, 71, 46, 75, 77, 52, 60, 10, 50, 53, 20, 5, 18, 
    #             89, 99, 86, 64, 65, 6, 21, 59, 95, 84, 66, 0, 41, 69, 45, 82, 55, 
    #             51, 19, 40, 9, 54, 23, 48, 35, 22, 2, 62]
    #     order = [o + 1 for o in order]

    #     jac_cc = jax.jit(jacve(RoeFlux_1d, order=order, argnums=(0, 1, 2, 3, 4, 5), count_ops=True))
    #     veres, aux = jac_cc(*xs)
                
    #     jaxpr = jax.make_jaxpr(RoeFlux_1d)(*xs)
    #     deriv_jaxpr = jax.make_jaxpr(jacve(RoeFlux_1d, order=order, argnums=(0, 1, 2, 3, 4, 5)))(*xs)
    #     print(deriv_jaxpr)
    #     print("num_muls", aux["num_muls"])
    #     print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

    #     jax_jac_rev = jax.jit(jax.jacrev(RoeFlux_1d, argnums=(0, 1, 2, 3, 4, 5)))
    #     revres = jax_jac_rev(*xs)
    #     print(veres)
    #     print(revres)
    #     self.assertTrue(tree_allclose(veres, revres))          
        
    def test_EncoderDecoder(self):
        x = jnp.ones((4, 4))
        y = jnp.ones((2, 4))

        WQ1 = jnp.ones((4, 4))
        WK1 = jnp.ones((4, 4))
        WV1 = jnp.ones((4, 4))

        WQ2 = jnp.ones((4, 4))
        WK2 = jnp.ones((4, 4))
        WV2 = jnp.ones((4, 4))

        WQ3 = jnp.ones((4, 4))
        WK3 = jnp.ones((4, 4))
        WV3 = jnp.ones((4, 4))

        W1 = jnp.ones((4, 4))
        b1 = jnp.ones(4)

        W2 = jnp.ones((2, 4))
        b2 = jnp.ones((2, 1))

        xs = (x, y, WQ1, WQ2, WQ3, WK1, WK2, WK3, WV1, WV2, WV3, W1, W2, b1, b2, 0., 1., 0., 1., 0., 1.)

        # order = [65, 33, 26, 122, 75, 13, 58, 35, 38, 53, 31, 40, 104, 73, 92, 
        #         79, 106, 89, 52, 39, 72, 125, 101, 24, 78, 90, 110, 56, 116, 42,
        #         87,  16, 76, 123, 114, 99, 1, 28, 70, 105, 121, 120, 115, 61, 71, 
        #         124, 9, 109, 14, 112, 82, 41, 97, 91, 88, 67, 95, 68, 51, 2, 25,
        #         84, 17, 118, 0, 80, 64, 23, 113, 11, 12, 63, 81, 111, 49, 20, 77,
        #         117, 29, 86, 54, 45, 107, 37, 21, 126, 48, 69, 4, 98, 57, 32, 36,
        #         8, 6, 102, 83, 103, 62, 55, 47, 10, 59, 19, 46, 30, 94, 85, 5, 74,
        #         7, 93, 15, 119, 108, 100, 27, 22, 44, 3, 34, 66]
        
        # order = [o + 1 for o in order]
        order = "rev"
        
        jaxpr = jax.make_jaxpr(EncoderDecoder)(*xs)
        print(jaxpr)

        argnums = range(len(xs))
        jac_cc = jax.jit(jacve(EncoderDecoder, order=order, argnums=argnums, count_ops=True))
        veres, aux = jac_cc(*xs)
                
        jaxpr = jax.make_jaxpr(EncoderDecoder)(*xs)
        deriv_jaxpr = jax.make_jaxpr(jacve(EncoderDecoder, order=order, argnums=argnums))(*xs)
        print(deriv_jaxpr)
        print("num_muls", aux["num_muls"])
        print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

        jax_jac_rev = jax.jit(jax.jacrev(EncoderDecoder, argnums=argnums))
        revres = jax_jac_rev(*xs)
        self.assertTrue(tree_allclose(veres, revres))
        
    # def test_RobotArm(self):
    #     xs = [.02]*6
    #     order = [95, 58, 8, 36, 20, 45, 104, 18, 63, 6, 9, 64, 106, 94, 21, 93, 
    #             79, 76, 5, 78, 62, 13, 102, 89, 88, 77, 31, 66, 52, 55, 57, 80, 
    #             90, 87, 60, 51, 27, 25, 92, 112, 39, 29, 33, 75, 47, 68, 103, 50,
    #             98, 107, 49, 86, 16, 83, 91, 1, 96, 69, 44, 4, 19, 43, 28, 73, 
    #             108, 81, 10, 7, 37, 54, 105, 110, 70, 22, 3, 26, 34, 35, 99, 72, 
    #             17, 38, 30, 97, 40, 32, 85, 24, 82, 111, 42, 46, 59, 53, 100, 12,
    #             109, 14, 74, 15, 56, 41, 48, 0, 2, 71, 11, 23]
    #     order = [o + 1 for o in order]
        
    #     mM_order = [6, 7, 9, 10, 14, 18, 19, 21, 22, 53, 63, 64, 65, 67, 77, 78, 
    #                 79, 80, 82, 94, 95, 96, 97, 99, 103, 104, 105, 107, 113, 2, 
    #                 5, 8, 11, 17, 20, 23, 25, 27, 29, 30, 33, 35, 37, 38, 41, 43, 
    #                 45, 46, 49, 50, 54, 55, 58, 59, 69, 71, 73, 74, 81, 84, 86, 
    #                 88, 90, 91, 98, 101, 106, 109, 111, 26, 31, 34, 39, 42, 47, 
    #                 51, 56, 60, 70, 75, 83, 87, 92, 100, 108, 110, 52, 61, 15, 
    #                 28, 36, 44, 48, 57, 72, 1, 89, 13, 112, 3, 32, 40, 4, 16, 76, 
    #                 93, 12, 24]


    #     jac_cc = jax.jit(jacve(RobotArm_6DOF, order=order, argnums=(0, 1, 2, 3, 4, 5), count_ops=True))
    #     veres, aux = jac_cc(*xs)
                
    #     deriv_jaxpr = jax.make_jaxpr(jacve(RobotArm_6DOF, order=order, argnums=(0, 1, 2, 3, 4, 5)))(*xs)
    #     # print(deriv_jaxpr)
    #     print("CC num_muls", aux["num_muls"])
        
        
    #     jac_mM = jax.jit(jacve(RobotArm_6DOF, order=mM_order, argnums=(0, 1, 2, 3, 4, 5), count_ops=True))
    #     _, aux = jac_mM(*xs)
        
    #     print("mM num_muls", aux["num_muls"])
        
    #     jac_rev = jax.jit(jacve(RobotArm_6DOF, order="rev", argnums=(0, 1, 2, 3, 4, 5), count_ops=True))
    #     _, aux = jac_rev(*xs)
        
    #     print("rev num_muls", aux["num_muls"])
                
    #     jaxpr = jax.make_jaxpr(RobotArm_6DOF)(*xs)
    #     deriv_jaxpr = jax.make_jaxpr(jacve(RobotArm_6DOF, order=order, argnums=(0, 1, 2, 3, 4, 5)))(*xs)
    #     # print(deriv_jaxpr)
    #     print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

    #     jax_jac_rev = jax.jit(jax.jacrev(RobotArm_6DOF, argnums=(0, 1, 2, 3, 4, 5)))
    #     revres = jax_jac_rev(*xs)
    #     print(veres)
    #     print(revres)
    #     self.assertTrue(tree_allclose(veres, revres))    
        
    # def test_Lighthouse(self):
    #     xs = [.02]*4
    #     order = [1, 2, 3, 4, 5, 6, 7][::-1]
        
    #     print(jax.make_jaxpr(Lighthouse)(*xs))

    #     jac_cc = jax.jit(jacve(Lighthouse, order=order, argnums=(0, 1, 2, 3), count_ops=True))
    #     veres, aux = jac_cc(*xs)
                
    #     deriv_jaxpr = jax.make_jaxpr(jacve(Lighthouse, order=order, argnums=(0, 1, 2, 3)))(*xs)
    #     print(deriv_jaxpr)
    #     print("CC num_muls", aux["num_muls"])
                        
    #     jaxpr = jax.make_jaxpr(Lighthouse)(*xs)
    #     deriv_jaxpr = jax.make_jaxpr(jacve(Lighthouse, order=order, argnums=(0, 1, 2, 3)))(*xs)
    #     print(deriv_jaxpr)
    #     print(count_muls_jaxpr(deriv_jaxpr) - count_muls_jaxpr(jaxpr))

    #     jax_jac_rev = jax.jit(jax.jacrev(Lighthouse, argnums=(0, 1, 2, 3)))
    #     revres = jax_jac_rev(*xs)
    #     print(veres)
    #     print(revres)
    #     self.assertTrue(tree_allclose(veres, revres))  


if __name__ == '__main__':
    unittest.main()

