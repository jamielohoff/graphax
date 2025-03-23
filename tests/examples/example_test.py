import unittest
from functools import partial
from typing import Callable, Sequence

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand


from graphax import jacve, tree_allclose
from graphax.examples import (Simple, Helmholtz, f, g, RoeFlux_1d, RobotArm_6DOF,
                              EncoderDecoder, Lighthouse, RoeFlux_3d, Perceptron,
                              Encoder, BlackScholes_Jacobian)


def test_order(order: str | Sequence[int], fn: Callable, argnums: Sequence[int],
               *args) -> bool:
    jacve_f = jax.jit(jacve(fn, order=order, argnums=argnums, count_ops=True))
    veres, aux = jacve_f(*args)
    print("num muls:", aux["num_muls"])
            
    jacrev_f = jax.jit(jax.jacrev(fn, argnums=argnums))
    revres = jacrev_f(*args)

    return tree_allclose(veres, revres)

test_rev = partial(test_order, "rev")

def test_fwd(fn: Callable, argnums: Sequence[int],*args) -> bool:
    jacve_f = jax.jit(jacve(fn, order="fwd", argnums=argnums, count_ops=True))
    veres, aux = jacve_f(*args)
    print("num muls:", aux["num_muls"])

    jacfwd_f = jax.jit(jax.jacrev(fn, argnums=argnums))
    fwdres = jacfwd_f(*args)

    return tree_allclose(veres, fwdres)


class ExampleTests(unittest.TestCase): 
    def test_Simple(self):
        print("Testing Simple()...")
        args = (5., 7.)
        
        self.assertTrue(test_fwd(Simple, (0, 1), *args))
        self.assertTrue(test_rev(Simple, (0, 1), *args))

        x = jnp.ones((50, 50))
        y = jnp.ones((50, 50))
        args = (x, y)

        self.assertTrue(test_fwd(Simple, (0, 1), *args))
        self.assertTrue(test_rev(Simple, (0, 1), *args))

    def test_Lighthouse(self):
        print("Testing Lighthouse()...")
        xs = [.02]*4
        argnums = (0, 1, 2, 3)

        self.assertTrue(test_fwd(Lighthouse, argnums, *xs))
        self.assertTrue(test_rev(Lighthouse, argnums, *xs))

    def test_Helmholtz(self):
        print("Testing Helmholtz()...")
        xs = jnp.array([0.05, 0.15, 0.25, 0.35])
        order = [2, 5, 4, 3, 1]

        args =(xs,)

        self.assertTrue(test_fwd(Helmholtz, (0,), *args))
        self.assertTrue(test_rev(Helmholtz, (0,), *args))
        self.assertTrue(test_order(order, Helmholtz, (0,), *args))

    def test_RobotArm_6DOF(self):
        print("Testing RobotArm_6DOF()...")
        args = [.02]*6
        argnums = range(len(args))

        order = [95, 58, 8, 36, 20, 45, 104, 18, 63, 6, 9, 64, 106, 94, 21, 93, 
                79, 76, 5, 78, 62, 13, 102, 89, 88, 77, 31, 66, 52, 55, 57, 80, 
                90, 87, 60, 51, 27, 25, 92, 112, 39, 29, 33, 75, 47, 68, 103, 50,
                98, 107, 49, 86, 16, 83, 91, 1, 96, 69, 44, 4, 19, 43, 28, 73, 
                108, 81, 10, 7, 37, 54, 105, 110, 70, 22, 3, 26, 34, 35, 99, 72, 
                17, 38, 30, 97, 40, 32, 85, 24, 82, 111, 42, 46, 59, 53, 100, 12,
                109, 14, 74, 15, 56, 41, 48, 0, 2, 71, 11, 23]
        order = [o + 1 for o in order]
        
        mM_order = [6, 7, 9, 10, 14, 18, 19, 21, 22, 53, 63, 64, 65, 67, 77, 78, 
                    79, 80, 82, 94, 95, 96, 97, 99, 103, 104, 105, 107, 113, 2, 
                    5, 8, 11, 17, 20, 23, 25, 27, 29, 30, 33, 35, 37, 38, 41, 43, 
                    45, 46, 49, 50, 54, 55, 58, 59, 69, 71, 73, 74, 81, 84, 86, 
                    88, 90, 91, 98, 101, 106, 109, 111, 26, 31, 34, 39, 42, 47, 
                    51, 56, 60, 70, 75, 83, 87, 92, 100, 108, 110, 52, 61, 15, 
                    28, 36, 44, 48, 57, 72, 1, 89, 13, 112, 3, 32, 40, 4, 16, 76, 
                    93, 12, 24]
        
        self.assertTrue(test_fwd(RobotArm_6DOF, argnums, *args))
        self.assertTrue(test_rev(RobotArm_6DOF, argnums, *args))
        self.assertTrue(test_order(order, RobotArm_6DOF, argnums, *args))
        self.assertTrue(test_order(mM_order, RobotArm_6DOF, argnums, *args))

    def test_RoeFlux_1d(self):
        print("Testing RoeFlux_1d()...")
        args = (.01, .02, .02, .01, .03, .03)
        argnums = (0, 1, 2, 3, 4, 5)
        order = [95, 7, 26, 16, 3, 49, 91, 37, 83, 88, 32, 68, 44, 81, 66, 24, 
                76, 85, 43, 86, 80, 42, 12, 15, 30, 62, 52, 78, 70, 58, 72, 56, 
                39, 94, 47, 10, 90, 46, 99, 1, 25, 41, 28, 71, 36, 57, 31, 21, 
                27, 8, 5, 33, 89, 84, 59, 20, 77, 73, 87, 75, 53, 97, 93, 64, 18, 
                45, 13, 74, 67, 79, 63, 60, 0, 48, 4, 65, 50, 92, 17, 6, 19, 9, 
                69, 55, 61, 82, 51, 40, 14, 35, 54, 38, 22, 2, 23, 11, 34, 29]
                
        order = [o + 1 for o in order]
        
        self.assertTrue(test_fwd(RoeFlux_1d, argnums, *args))
        self.assertTrue(test_rev(RoeFlux_1d, argnums, *args))
        self.assertTrue(test_order(order, RoeFlux_1d, argnums, *args))
        
    def test_RoeFlux_3d(self):
        print("Testing RoeFlux_3d()...")
        ul0 = jnp.array([.1])
        ul = jnp.array([.1, .2, .3])
        ul4 = jnp.array([.5])
        ur0 = jnp.array([.2])
        ur = jnp.array([.2, .2, .4])
        ur4 = jnp.array([.6])
        args = (ul0, ul, ul4, ur0, ur, ur4)
        argnums = list(range(len(args)))

        order = [124, 136, 56, 128, 78, 24, 1, 54, 101, 127, 121, 140, 47, 135, 
                67, 34, 111, 32, 100, 119, 99, 114, 125, 141, 122, 45, 65, 59, 
                117, 89, 116, 60, 42, 28, 74, 85, 11, 53, 36, 30, 108, 113, 55, 
                109, 129, 64, 91, 14, 133, 5, 10, 132, 87, 139, 110, 12, 131, 
                72, 8, 61, 88, 107, 6, 29, 57, 96, 118, 105, 71, 77, 112, 66, 
                75, 84, 143, 123, 90, 94, 137, 104, 69, 23, 22, 62, 58, 50, 
                130, 31, 106, 39, 48, 49, 98, 134, 93, 138, 126, 68, 115, 80, 
                102, 92, 79, 52, 16, 120, 95, 76, 19, 25, 73, 21, 70, 38, 35, 
                20, 86, 41, 4, 103, 43, 27, 3, 40, 9, 83, 13, 18, 37, 51, 46, 
                7, 81, 97, 63, 44, 2, 33, 82, 26, 15, 17, 145] 

        self.assertTrue(test_fwd(RoeFlux_3d, argnums, *args))
        self.assertTrue(test_rev(RoeFlux_3d, argnums, *args))
        # TODO: order is deprecated!
        # self.assertTrue(test_order(order, RoeFlux_3d, argnums, *args))

        batchsize = 16

        order = [77, 100, 133, 107, 129, 137, 5, 19, 95, 28, 37, 14, 135, 85, 
                51, 10, 115, 128, 63, 43, 9, 83, 104, 45, 99, 98, 39, 57, 108, 
                40, 82, 84, 22, 21, 32, 126, 38, 68, 67, 55, 97, 101, 53, 52, 
                27, 44, 94, 31, 7, 103, 131, 30, 12, 70, 69, 65, 87, 109, 122, 
                29, 6, 11, 64, 105, 102, 41, 3, 92, 33, 16, 13, 88, 73, 4, 61, 
                56, 91, 54, 72, 86, 121, 120, 118, 93, 75, 81, 111, 110, 125, 
                130, 47, 116, 66, 50, 25, 26, 59, 96, 49, 119, 35, 62, 8, 117, 
                15, 114, 89, 48, 76, 127, 78, 74, 124, 112, 123, 113, 106, 71, 
                46, 18, 58, 1, 36, 80, 79, 42, 60, 20, 17, 2, 132, 90, 34, 23, 
                24, 134, 136, 138, 139, 140, 141, 143, 145] 

        args = [jnp.tile(arg[jnp.newaxis, ...], (batchsize, 1)) for arg in args]
        vmap_RoeFlux_3d = jax.vmap(RoeFlux_3d)

        self.assertTrue(test_fwd(vmap_RoeFlux_3d, argnums, *args))
        self.assertTrue(test_rev(vmap_RoeFlux_3d, argnums, *args))
        self.assertTrue(test_order(order, vmap_RoeFlux_3d, argnums, *args))

    def test_NeuralNetwork(self):
        print("Testing NeuralNetwork()...")
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

        args = (x, W1, b1, W2, b2, y)
        argnums = (1, 2, 3, 4)

        self.assertTrue(test_fwd(NeuralNetwork, argnums, *args))
        self.assertTrue(test_rev(NeuralNetwork, argnums, *args))
    
    def test_vmap_NeuralNetwork(self):
        print("Testing vmap_NeuralNetwork()...")
        # TODO fix this unit test
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

        args = (x, W1, b1, W2, b2, y)
        argnums = (1, 2, 3, 4)

        self.assertTrue(test_fwd(NeuralNetwork, argnums, *args))
        self.assertTrue(test_rev(NeuralNetwork, argnums, *args))

        self.assertTrue(test_fwd(f, argnums, *args))
        self.assertTrue(test_rev(f, argnums, *args))

    def test_Perceptron(self):
        print("Testing Perceptron()...")
        key = jrand.PRNGKey(1234)

        x = jnp.ones(4)
        y = jrand.normal(key, (4,))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))

        args = (x, y, W1, b1, W2, b2, 0., 1.)
        
        argnums = list(range(len(args)))
        
        self.assertTrue(test_fwd(Perceptron, argnums, *args))
        self.assertTrue(test_rev(Perceptron, argnums, *args))

    def test_vmap_Perceptron(self):
        print("Testing vmap_Perceptron()...")
        key = jrand.PRNGKey(1234)

        x = jnp.ones((16, 4))
        y = jrand.normal(key, (16, 4))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8, 4))
        b1 = jrand.normal(b1key, (8,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4, 8))
        b2 = jrand.normal(b2key, (4,))

        args = (x, y, W1, b1, W2, b2, 0., 1.)
        vmap_Perceptron = jax.vmap(Perceptron, in_axes=(0, 0, None, None, None, None, None, None))
        argnums = list(range(len(args)))
        
        self.assertTrue(test_fwd(vmap_Perceptron, argnums, *args))
        self.assertTrue(test_rev(vmap_Perceptron, argnums, *args))
        
    def test_Encoder(self):
        print("Testing Encoder()...")
        key = jrand.PRNGKey(250197)
        x = jnp.ones((4, 4))
        y = jrand.normal(key, (2, 4))

        wq1key, wk1key, wv1key, key = jrand.split(key, 4)
        WQ1 = jrand.normal(wq1key, (4, 4))
        WK1 = jrand.normal(wk1key, (4, 4))
        WV1 = jrand.normal(wv1key, (4, 4))

        wq2key, wk2key, wv2key, key = jrand.split(key, 4)
        WQ2 = jrand.normal(wq2key, (4, 4))
        WK2 = jrand.normal(wk2key, (4, 4))
        WV2 = jrand.normal(wv2key, (4, 4))

        w1key, w2key, b1key, b2key = jrand.split(key, 4)
        W1 = jrand.normal(w1key, (4, 4))
        b1 = jrand.normal(b1key, (4,))

        W2 = jrand.normal(w2key, (2, 4))
        b2 = jrand.normal(b2key, (2, 1))
        
        args = (x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 0., 1., 0., 1.)
        
        argnums = list(range(len(args)))[2:]

        order = [50, 32, 84, 7, 19, 4, 49, 83, 9, 51, 92, 89, 37, 64, 72, 31, 
                91, 77, 35, 93, 40, 23, 69, 79, 48, 70, 58, 82, 62, 86, 21, 
                29, 22, 80, 73, 67, 28, 13, 90, 85, 30, 45, 71, 20, 39, 8, 24, 
                78, 10, 38, 36, 54, 60, 81, 27, 76, 25, 52, 17, 68, 61, 47, 65, 
                66, 88, 87, 11, 6, 75, 12, 74, 53, 63, 59, 57, 56, 55, 44, 46, 
                43, 42, 41, 34, 33, 26, 18, 16, 15, 14, 5, 3, 2, 1]
        
        self.assertTrue(test_fwd(Encoder, argnums, *args))
        self.assertTrue(test_rev(Encoder, argnums, *args))
        self.assertTrue(test_order(order, Encoder, argnums, *args))

    def test_EncoderDecoder(self):
        print("Testing EncoderDecoder()...")
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

        args = (x, y, WQ1, WQ2, WQ3, WK1, WK2, WK3, WV1, WV2, WV3, W1, W2, b1, b2, 0., 1., 0., 1., 0., 1.)
        argnums = range(len(args))[2:]

        self.assertTrue(test_fwd(EncoderDecoder, argnums, *args))
        self.assertTrue(test_rev(EncoderDecoder, argnums, *args))
        
    def test_BlackScholes_Jacobian(self):
        print("Testing BlackScholes_Jacobian()...")
        args = [1.]*5
        argnums = list(range(len(args)))
        
        self.assertTrue(test_fwd(BlackScholes_Jacobian, argnums, *args))
        self.assertTrue(test_rev(BlackScholes_Jacobian, argnums, *args))

        args = [jnp.ones((16,)) for _ in range(5)]

        vmap_BSJ = jax.vmap(BlackScholes_Jacobian)
        self.assertTrue(test_fwd(vmap_BSJ, argnums, *args))
        self.assertTrue(test_rev(vmap_BSJ, argnums, *args))

    # def test_f(self):
    #     print("Testing f()...")
    #     key = jrand.PRNGKey(42)
    #     a = jrand.uniform(key, (4,))
    #     b = jrand.uniform(key, (2, 3))
    #     c = jrand.uniform(key, (4, 4))
    #     d = jrand.uniform(key, (4, 1))
    #     args = [a, b, c, d]
    #     argnums = list(range(len(args)))
        
    #     # TODO this order is outdated as the function has changed slightly
    #     order = [33, 8, 16, 77, 15, 62, 40, 58, 14, 76, 42, 60, 54, 34, 61, 72, 
    #             37, 55, 18, 75, 36, 74, 65, 26, 35, 25, 66, 38, 64, 59, 53, 20, 
    #             27, 47, 10, 69, 23, 11, 41, 79, 9, 7, 12, 63, 71, 24, 67, 51, 4, 
    #             1, 21, 3, 6, 2, 49, 13, 44, 46, 56, 17, 39, 57, 43, 32, 52, 30, 
    #             48, 31, 5, 22, 45, 19, 50, 28, 29] 

    #     self.assertTrue(test_fwd(f, argnums, *args))
    #     self.assertTrue(test_rev(f, argnums, *args))
    #     # self.assertTrue(test_order(order, f, argnums, *args))
    
    # def test_g(self):
    #     print("Testing g()...")
    #     xs = [.15]*15
    #     argnums = list(range(len(xs)))
        
    #     self.assertTrue(test_fwd(g, argnums, *xs))
    #     self.assertTrue(test_rev(g, argnums, *xs))
        

if __name__ == "__main__":
    unittest.main()

