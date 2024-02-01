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
        
    # def test_softmax_0(self):
    #     def f(x, y):
    #         return jnp.sin(jnn.softmax(x @ y, axis=0))
        
    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))

    #     print(jax.make_jaxpr(f)(x, y))
    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)
        
    #     print("ve", veres[0])
    #     print("jax", revres[0])

    #     self.assertTrue(tree_allclose(veres, revres))        
        
    # def test_softmax_1(self):
    #     def f(x, y):
    #         return jnp.sin(jnn.softmax(x @ y, axis=1))
    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))

    #     print(jax.make_jaxpr(f)(x, y))
    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(f, argnums=(0, 1))(x, y)
        
    #     print(veres[1])
    #     print(revres[1])

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

    #     deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0, 1, 2, 3, 4)))
    #     veres = deriv_fn(*xs)

    #     revres = jax.jacrev(f, argnums=(0, 1, 2, 3, 4))(*xs)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_cross_entropy(self):
    #     def log_softmax(x):
            
    #     def ce_loss(x, y):
    #         return optax.softmax_cross_entropy(x, y)

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (32,))
    #     y = jrand.normal(ykey, (32,))
        
    #     print(jax.make_jaxpr(ce_loss)(x, y))
        
    #     deriv_fn = jax.jit(jacve(ce_loss, order="fwd", argnums=(0, )))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(ce_loss, argnums=(0,))(x, y)
        
    #     print(veres)
    #     print(revres)

    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_softmax_self_attention_fwd(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         return jnn.softmax(a, axis=0) @ v
        
    #     key = jrand.PRNGKey(42)
    #     xkey, qkey, kkey, vkey = jrand.split(key, 4)
    #     s = 10
    #     x = jrand.uniform(xkey, (s, 2*s), minval=.5, maxval=1.5)
    #     WQ = jrand.uniform(qkey, (s, s), minval=.5, maxval=1.5)
    #     WK = jrand.uniform(kkey, (s, s), minval=.5, maxval=1.5)
    #     WV = jrand.uniform(vkey, (s, s), minval=.5, maxval=1.5)

    #     print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacfwd(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacfwd(softmax_attention, argnums=(1, 2, 3)))
    #     revres = jax_jac_rev(x, WQ, WK, WV)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))
    #     veres = jac_rev(x, WQ, WK, WV)
        
    #     print("ve", veres[0].sum())
    #     print("jax", revres[0].sum())
        
    #     print("ve", veres[1].sum())
    #     print("jax", revres[1].sum())
        
    #     print("ve", veres[2].sum())
    #     print("jax", revres[2].sum())

    #     self.assertTrue(tree_allclose(veres, revres))

    # def test_softmax_self_attention_rev(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         return jnn.softmax(a, axis=0) @ v
        
    #     key = jrand.PRNGKey(42)
    #     xkey, qkey, kkey, vkey = jrand.split(key, 4)
    #     s = 10
    #     x = jrand.uniform(xkey, (s, 2*s), minval=.5, maxval=1.5)
    #     WQ = jrand.uniform(qkey, (s, s), minval=.5, maxval=1.5)
    #     WK = jrand.uniform(kkey, (s, s), minval=.5, maxval=1.5)
    #     WV = jrand.uniform(vkey, (s, s), minval=.5, maxval=1.5)

    #     print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))
    #     revres = jax_jac_rev(x, WQ, WK, WV)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))
    #     veres = jac_rev(x, WQ, WK, WV)
        
    #     print("ve", veres[0].sum())
    #     print("jax", revres[0].sum())
        
    #     print("ve", veres[1].sum())
    #     print("jax", revres[1].sum())
        
    #     print("ve", veres[2].sum())
    #     print("jax", revres[2].sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_multihead_attention(self):
    #     num_heads = 6
    #     dk = 10
    #     embedding_dim = 15
    #     seq_len = 20
    #     def multihead_softmax_attention(X, WQ, WK, WV, WO):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         out = jnn.softmax(a, axis=0) @ v
    #         return WO @ out 
        
    #     x = jnp.arange(embedding_dim*seq_len, dtype=jnp.float32).reshape(embedding_dim, seq_len)
    #     WQ = .2*jnp.arange(embedding_dim*dk*num_heads, dtype=jnp.float32).reshape(dk*num_heads, embedding_dim) 
    #     WK = .33*jnp.arange(embedding_dim*dk*num_heads, dtype=jnp.float32)[::-1].reshape(dk*num_heads, embedding_dim)
    #     WV = .5*jnp.arange(embedding_dim*dk*num_heads, dtype=jnp.float32).reshape(dk*num_heads, embedding_dim)
    #     WO = .71*jnp.arange(embedding_dim*dk*num_heads, dtype=jnp.float32).reshape(embedding_dim, dk*num_heads)

    #     print(jax.make_jaxpr(multihead_softmax_attention)(x, WQ, WK, WV, WO))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(multihead_softmax_attention, argnums=(1, 2, 3, 4)))
    #     revres = jax_jac_rev(x, WQ, WK, WV, WO)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(multihead_softmax_attention, order="rev", argnums=(1, 2, 3, 4)))
    #     veres = jac_rev(x, WQ, WK, WV, WO)
        
    #     print("ve", veres[0].sum())
    #     print("jax", revres[0].sum())
        
    #     print("ve", veres[1].sum())
    #     print("jax", revres[1].sum())
        
    #     print("ve", veres[2].sum())
    #     print("jax", revres[2].sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
    
    def test_vmap_multihead_attention(self):
        batchsize = 16
        num_heads = 6
        dk = 10
        embedding_dim = 15
        seq_len = 20
        
        @partial(jax.vmap, in_axes=(0, None, None, None, None))
        def multihead_softmax_attention(X, WQ, WK, WV, WO):
            q = WQ @ X
            k = WK @ X
            v = WV @ X
            a = q @ k.T
            out = jnn.softmax(a, axis=0) @ v
            return WO @ out 
        
        # Weigths for self-attention layer
        key = jrand.PRNGKey(42)
        xkey, qkey, kkey, vkey, okey = jrand.split(key, 5)
        X = jrand.normal(xkey, (batchsize, embedding_dim, seq_len))
        WQ = jrand.normal(qkey, (dk*num_heads, embedding_dim))
        WK = jrand.normal(kkey, (dk*num_heads, embedding_dim))
        WV = jrand.normal(vkey, (dk*num_heads, embedding_dim))
        WO = jrand.normal(okey, (embedding_dim, dk*num_heads))

        print(jax.make_jaxpr(multihead_softmax_attention)(X, WQ, WK, WV, WO))

        # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jac_rev = jax.jit(jacve(multihead_softmax_attention, order="rev", argnums=(1, 2, 3, 4)))
        
        # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jax_jac_rev = jax.jit(jax.jacrev(multihead_softmax_attention, argnums=(1, 2, 3, 4)))
        revres = jax_jac_rev(X, WQ, WK, WV, WO)
        veres = jac_rev(X, WQ, WK, WV, WO)
        
        print("ve", veres[0].sum())
        print("jax", revres[0].sum())
        
        print("ve", veres[1].sum())
        print("jax", revres[1].sum())
        
        print("ve", veres[2].sum())
        print("jax", revres[2].sum())
        
        self.assertTrue(tree_allclose(veres, revres))
        
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
        
        
    #     self.assertTrue(tree_allclose(veres, revres))
    
    # def test_two_layer_vmap_multihead_transformer(self):
    #     batchsize = 16
    #     num_heads = 6
    #     dk = 10
    #     embedding_dim = 15
    #     seq_len = 20
    #     def multihead_softmax_attention(X, WQ, WK, WV, WO):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         out = jnn.softmax(a, axis=0) @ v
    #         return WO @ out 
        
    #     # Weigths for first self-attention layer
    #     key = jrand.PRNGKey(42)
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ1 = jrand.normal(qkey, (dk*num_heads, embedding_dim))
    #     WK1 = jrand.normal(kkey, (dk*num_heads, embedding_dim))
    #     WV1 = jrand.normal(vkey, (dk*num_heads, embedding_dim))
    #     WO1 = jrand.normal(okey, (embedding_dim, dk*num_heads))
        
    #     # Weights for second self-attention layer
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ2 = jrand.normal(qkey, (dk*num_heads, embedding_dim))
    #     WK2 = jrand.normal(kkey, (dk*num_heads, embedding_dim))
    #     WV2 = jrand.normal(vkey, (dk*num_heads, embedding_dim))
    #     WO2 = jrand.normal(okey, (embedding_dim, dk*num_heads))
        
    #     def MLP(X, W1, b1, W2, b2):
    #         out = jnp.tanh(W1 @ X + b1)
    #         return jnp.tanh(W2 @ out + b2)
        
    #     W1key, b1key, W2key, b2key, key = jrand.split(key, 5)
    #     W1 = jrand.normal(W1key, (10, embedding_dim))
    #     b1 = jrand.normal(b1key, (10, 1))
    #     W2 = jrand.normal(W2key, (embedding_dim, 10))
    #     b2 = jrand.normal(b2key, (embedding_dim, 1))
        
    #     W3key, b3key, W4key, b4key, key = jrand.split(key, 5)
    #     W3 = jrand.normal(W3key, (10, embedding_dim))
    #     b3 = jrand.normal(b3key, (10, 1))
    #     W4 = jrand.normal(W4key, (5, 10))
    #     b4 = jrand.normal(b4key, (5, 1))
        
    #     @partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, 
    #                                 None, None, None, None, None, None, None, None, 0))
    #     def transformer(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
    #                     W1, b1, W2, b2, W3, b3, W4, b4, label):
    #         X = multihead_softmax_attention(X, WQ1, WK1, WV1, WO1)
    #         X = MLP(X, W1, b1, W2, b2)
    #         X = multihead_softmax_attention(X, WQ2, WK2, WV2, WO2)
    #         X = MLP(X, W3, b3, W4, b4)
    #         return .5*(X-label)**2
        
    #     def model(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
    #                 W1, b1, W2, b2, W3, b3, W4, b4, label):
    #         return transformer(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
    #                             W1, b1, W2, b2, W3, b3, W4, b4, label).sum()
        
    #     X = jrand.normal(key, (batchsize, embedding_dim, seq_len))
    #     label = jnp.arange(batchsize*5*20, dtype=jnp.float32).reshape(batchsize, 5, seq_len)
        
    #     print(jax.make_jaxpr(model)(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
    #                                         W1, b1, W2, b2, W3, b3, W4, b4, label))
        
    #     print(model(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, W1, b1, W2, b2, W3, b3, W4, b4, label))
        
    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(model, argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
    #                                                     11, 12, 13, 14, 15, 16, 17)))
    #     revres = jax_jac_rev(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, W1, b1, W2, b2, W3, b3, W4, b4, label)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(model, order="rev", argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    #                                                          11, 12, 13, 14, 15, 16, 17)))
    #     veres = jac_rev(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, W1, b1, W2, b2, W3, b3, W4, b4, label)
        
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
        


if __name__ == '__main__':
    unittest.main()

