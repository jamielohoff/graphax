import unittest

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.examples import Simple, Helmholtz, f, g


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

    # def test_transpose(self):
    #     def transpose(x, y):
    #         return x.T + y

    #     x = jnp.ones((2, 3))
    #     y = jnp.ones((3, 2))
    #     jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
    #     veres = jac_fwd(x, y)[0]

    #     revres = jax.jacrev(transpose)(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_slicing(self):
    #     def f(x, y):
    #         x = jnp.sin(x)
    #         x = lax.slice(x, start_indices=[0, 0], limit_indices=[2, 3])
    #         return x * y

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (3,3))
    #     y = jrand.normal(ykey, (3,))

    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)

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

    # def test_simple(self):
    #     x = jnp.ones((50, 50))
    #     y = jnp.ones((50, 50))

    #     jacrev_f = jax.jit(jacve(Simple, order="rev", argnums=(0, 1), count_ops=True))
    #     veres = jacrev_f(x, y)

    #     jac_f = jax.jit(jax.jacrev(Simple, argnums=(0, 1)))
    #     revres = jac_f(x, y)

    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_Helmholtz(self):
    #     x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(4)/2000. # 

    #     jac_cc = jax.jit(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))
    #     veres = jac_cc(x)

    #     jax_jac_fwd = jax.jit(jax.jacfwd(Helmholtz))
    #     jax_jac_rev = jax.jit(jax.jacfwd(Helmholtz))
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
        
    # def test_f(self):
    #     a = jnp.ones(4)
    #     b = jnp.ones((2, 3))
    #     c = jnp.ones((4, 4))
    #     d = jnp.ones((3, 3))
    #     e = jnp.ones((4, 1))
    #     xs = [a, b, c, d, e]

    #     deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0, 1, 2, 3, 4)))
    #     veres = deriv_fn(*xs)

    #     revres = jax.jacrev(f, argnums=(0, 1, 2, 3, 4))(*xs)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_softmax_attention_fwd(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         return jnn.softmax(a, axis=1) @ v
        
    #     x = jnp.arange(160, dtype=jnp.float32).reshape(10, 16)
    #     WQ = .2*jnp.arange(100, dtype=jnp.float32).reshape(10, 10) 
    #     WK = .33*jnp.arange(100, dtype=jnp.float32)[::-1].reshape(10, 10)
    #     WV = .5*jnp.arange(100, dtype=jnp.float32).reshape(10, 10)

    #     print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacfwd(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacfwd(softmax_attention, argnums=(1, 2, 3)))
    #     revres = jax_jac_rev(x, WQ, WK, WV)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))
    #     veres = jac_rev(x, WQ, WK, WV)
        
    #     print("ve", veres[2].sum())
    #     print("jax", revres[2].sum())

    #     self.assertTrue(tree_allclose(veres[0], revres[0]))

    def test_softmax_attention_rev(self):
        def softmax_attention(X, WQ, WK, WV):
            q = WQ @ X
            k = WK @ X
            v = WV @ X
            a = q @ k.T
            return jnn.softmax(a, axis=1) @ v
        
        x = jnp.arange(200, dtype=jnp.float32).reshape(10, 20)
        WQ = .2*jnp.arange(100, dtype=jnp.float32).reshape(10, 10) 
        WK = .33*jnp.arange(100, dtype=jnp.float32)[::-1].reshape(10, 10)
        WV = .5*jnp.arange(100, dtype=jnp.float32).reshape(10, 10)

        print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

        # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jax_jac_rev = jax.jit(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))
        revres = jax_jac_rev(x, WQ, WK, WV)

        # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jac_rev = jax.jit(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))
        veres = jac_rev(x, WQ, WK, WV)
        
        print("ve", veres[2].sum())
        print("jax", revres[2].sum())
        
        import time
        import matplotlib.pyplot as plt
        
        ve_data, jax_data = [], []
        for d in [1, 2, 4, 8]:
            x = jnp.arange(20*d*10*d, dtype=jnp.float32).reshape(10*d, 20*d)
            WQ = .2*jnp.arange(100*d**2, dtype=jnp.float32).reshape(10*d, 10*d) 
            WK = .33*jnp.arange(100*d**2, dtype=jnp.float32)[::-1].reshape(10*d, 10*d)
            WV = .5*jnp.arange(100*d**2, dtype=jnp.float32).reshape(10*d, 10*d)
            
            veres = jac_rev(x, WQ, WK, WV)
            revres = jax_jac_rev(x, WQ, WK, WV)
            
            st = time.time()
            for i in range(50):
                veres = jac_rev(x, WQ, WK, WV)
            ve_data.append((time.time() - st)/50)
            
            st = time.time()
            for i in range(50):
                revres = jax_jac_rev(x, WQ, WK, WV)
            jax_data.append((time.time() - st)/50)
            
        plt.plot([1, 2, 4, 8], ve_data, label="ve")
        plt.plot([1, 2, 4, 8], jax_data, label="jax")
        plt.legend()
        plt.imsave("cpu_softmax_attention.png", plt.gcf())

        self.assertTrue(tree_allclose(veres[0], revres[0]))
        
    # def test_two_layer_transformer(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         return jnn.softmax(a, axis=1) @ v
        
    #     def MLP(X, W, b):
    #         return jnp.tanh(W @ X + b)
        
    #     def transformer(X, WQ1, WK1, WV1, WQ2, WK2, WV2, W1, b1, W2, b2, label):
    #         X = softmax_attention(X, WQ1, WK1, WV1)
    #         X = MLP(X, W1, b1)
    #         X = softmax_attention(X, WQ2, WK2, WV2)
    #         X = MLP(X, W2, b2)
    #         return .5*(X-label)**2
        
    #     key = jrand.PRNGKey(42)
        
    #     x = jnp.arange(160, dtype=jnp.float32).reshape(10, 16)
    #     WQ1 = .2*jnp.arange(100, dtype=jnp.float32).reshape(10, 10) 
    #     WK1 = .33*jnp.arange(100, dtype=jnp.float32)[::-1].reshape(10, 10)
    #     WV1 = .5*jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
        
    #     W1 = jrand.normal(key, (10, 10))
    #     b1 = jrand.normal(key, (10, 1))
        
    #     WQ2 = .2*jnp.arange(100, dtype=jnp.float32).reshape(10, 10) 
    #     WK2 = .33*jnp.arange(100, dtype=jnp.float32)[::-1].reshape(10, 10)
    #     WV2 = .5*jnp.arange(100, dtype=jnp.float32).reshape(10, 10)
        
    #     W2 = jrand.normal(key, (5, 10))
    #     b2 = jrand.normal(key, (5, 1))
        
    #     label = jnp.arange(5*16, dtype=jnp.float32).reshape(5, 16)
        
    #     print(jax.make_jaxpr(transformer)(x, WQ1, WK1, WV1, WQ2, WK2, WV2, W1, b1, W2, b2, label))
        
    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(transformer, argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
    #     revres = jax_jac_rev(x, WQ1, WK1, WV1, WQ2, WK2, WV2, W1, b1, W2, b2, label)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(transformer, order="rev", argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))
    #     veres = jac_rev(x, WQ1, WK1, WV1, WQ2, WK2, WV2, W1, b1, W2, b2, label)
        
        
    #     self.assertTrue(True)
        
        


if __name__ == '__main__':
    unittest.main()

