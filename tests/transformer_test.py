import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

import numpy as np

from graphax import jacve, tree_allclose

from _transformer import (multihead_softmax_attention, MLP, layer_norm, glorot, 
                        make_positional_encoding, softmax_ce_loss, gelu,
                        multihead_attention_block)


positional_encoding = make_positional_encoding(32, 16)

### Transformer model
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, 
                            None, None, None, None, None, None, None, None, None,
                            None, None, None, None))
def transformer(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
                W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT):
    X = jnp.concatenate((CT, X), axis=1)
    X = positional_encoding(X)
    
    X = multihead_attention_block(X, WQ1, WK1, WV1, WO1, W1, b1, W2, b2)
    X = multihead_attention_block(X, WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
    
    X = X[:, 0]
    return W6 @ gelu(W5 @ X + b5) + b6





class TransformerTest(unittest.TestCase): 
    # ### Test of the utility building blocks
    # def test_cross_entropy(self):            

    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (32, 10))
    #     y = jrand.normal(ykey, (32, 10))
        
    #     print(jax.make_jaxpr(softmax_ce_loss)(x, y))
        
    #     deriv_fn = jax.jit(jacve(softmax_ce_loss, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     revres = jax.jacrev(softmax_ce_loss, argnums=(0, 1))(x, y)
        
    #     print(veres)
    #     print(revres)

    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_MLP(self):
    #     seq_len = 20
    #     embedding_dim = 15
        
    #     # Weights for MLP layers
    #     key = jrand.PRNGKey(42)
    #     W1key, W2key, key = jrand.split(key, 3)
    #     W1 = glorot(W1key, (10, embedding_dim))
    #     b1 = jnp.zeros((10, 1), dtype=jnp.float32)
    #     W2 = glorot(W2key, (embedding_dim, 10))
    #     b2 = jnp.zeros((embedding_dim, 1), dtype=jnp.float32)
        
    #     x = jrand.normal(key, (embedding_dim, seq_len))
        
    #     print(jax.make_jaxpr(MLP)(x, W1, b1, W2, b2))
        
    #     deriv_fn = jax.jit(jacve(MLP, order="rev", argnums=(1, 2, 3, 4)))
    #     veres = deriv_fn(x, W1, b1, W2, b2)

    #     revres = jax.jacrev(MLP, argnums=(1, 2, 3, 4))(x, W1, b1, W2, b2)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).sum())
    #     print("err2", jnp.abs(veres[1] - revres[1]).sum())
    #     print("err3", jnp.abs(veres[2] - revres[2]).sum())
    #     print("err4", jnp.abs(veres[3] - revres[3]).sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_multihead_attention_block(self):
    #     num_heads = 8
    #     seq_len = 32
    #     embedding_dim = 32
    #     dk = 32//num_heads

    #     ### Weights for self-attention layer
    #     key = jrand.PRNGKey(42)
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO = glorot(okey, (embedding_dim, dk*num_heads))
        
    #     # Weights for MLP layer
    #     W1key, W2key, key = jrand.split(key, 3)
    #     W1 = glorot(W1key, (1024, embedding_dim))
    #     b1 = jnp.zeros((1024, 1), dtype=jnp.float32)
    #     W2 = glorot(W2key, (embedding_dim, 1024))
    #     b2 = jnp.zeros((embedding_dim, 1), dtype=jnp.float32)
        
    #     x = jrand.normal(key, (embedding_dim, seq_len))
        
    #     print(jax.make_jaxpr(multihead_attention_block)(x, WQ, WK, WV, WO, W1, b1, W2, b2))
        
    #     argnums = range(1, 9)
    #     deriv_fn = jax.jit(jacve(multihead_attention_block, order="rev", argnums=argnums))
    #     veres = deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)

    #     jax_deriv_fn = jax.jit(jax.jacrev(multihead_attention_block, argnums=argnums))
    #     revres = jax_deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).sum())
    #     print("err2", jnp.abs(veres[1] - revres[1]).sum())
    #     print("err3", jnp.abs(veres[2] - revres[2]).sum())
    #     print("err4", jnp.abs(veres[3] - revres[3]).sum())
        
    #     print("err5", jnp.abs(veres[4] - revres[4]).sum())
    #     print("err6", jnp.abs(veres[5] - revres[5]).sum())
    #     print("err7", jnp.abs(veres[6] - revres[6]).sum())
    #     print("err8", jnp.abs(veres[7] - revres[7]).sum())
        
    #     import matplotlib.pyplot as plt
    #     import time
        
    #     out = jax.jit(deriv_fn)(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     st = time.time()
    #     for i in range(50):
    #         out = deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     print("graphax time", time.time() - st)
        
    #     jax_deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     st = time.time()
    #     for i in range(50):
    #         out = jax_deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     print("jax time", time.time() - st)
        
        # self.assertTrue(tree_allclose(veres, revres))
            
    # ### Test all the softmax attention crap
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
            
    # def test_softmax_self_attention_fwd(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T / jnp.sqrt(dk)
    #         return jnn.softmax(a, axis=0) @ v
        
    #     key = jrand.PRNGKey(42)
    #     xkey, qkey, kkey, vkey = jrand.split(key, 4)
    #     s = 10
    #     x = glorot(xkey, (s, 2*s))
    #     WQ = glorot(qkey, (s, s))
    #     WK = glorot(kkey, (s, s))
    #     WV = glorot(vkey, (s, s))

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
    #     seq_len = 32
    #     embedding_dim = 64
    #     dk = 64
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T / jnp.sqrt(dk)
    #         return jnn.softmax(a, axis=0) @ v
        
    #     key = jrand.PRNGKey(42)
    #     xkey, qkey, kkey, vkey = jrand.split(key, 4)
    #     x = jrand.normal(xkey, (embedding_dim, seq_len))
    #     WQ = glorot(qkey, (dk, embedding_dim))
    #     WK = glorot(kkey, (dk, embedding_dim))
    #     WV = glorot(vkey, (dk, embedding_dim))

    #     print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))
    #     revres = jax_jac_rev(x, WQ, WK, WV)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))
    #     veres = jac_rev(x, WQ, WK, WV)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).sum())
    #     print("err2", jnp.abs(veres[1] - revres[1]).sum())
    #     print("err3", jnp.abs(veres[2] - revres[2]).sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
    
    def test_multihead_attention(self):
        num_heads = 1
        seq_len = 32
        embedding_dim = 32
        dk = 32//num_heads
        
        key = jrand.PRNGKey(42)
        x = jrand.normal(key, (embedding_dim, seq_len))
        qkey, kkey, vkey, okey, key = jrand.split(key, 5)
        WQ = glorot(qkey, (dk*num_heads, embedding_dim))
        WK = glorot(kkey, (dk*num_heads, embedding_dim))
        WV = glorot(vkey, (dk*num_heads, embedding_dim))
        WO = glorot(okey, (embedding_dim, dk*num_heads))

        print(jax.make_jaxpr(multihead_softmax_attention)(x, WQ, WK, WV, WO))

        # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jax_jac_rev = jax.jit(jax.jacrev(multihead_softmax_attention, argnums=(1, 2, 3, 4)))
        revres = jax_jac_rev(x, WQ, WK, WV, WO)

        # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
        jac_rev = jax.jit(jacve(multihead_softmax_attention, order="rev", argnums=(1, 2, 3, 4)))
        veres = jac_rev(x, WQ, WK, WV, WO)
        
        print("err1", jnp.abs(veres[0] - revres[0]).sum())
        print("err2", jnp.abs(veres[1] - revres[1]).sum())
        print("err3", jnp.abs(veres[2] - revres[2]).sum())
        print("err4", jnp.abs(veres[3] - revres[3]).sum())
        
        self.assertTrue(tree_allclose(veres, revres))
            
    # def test_vmap_multihead_attention(self):
    #     batchsize = 16
    #     num_heads = 6
    #     dk = 10
    #     embedding_dim = 15
    #     seq_len = 20
        
    #     vmap_multihead_attn = jax.vmap(multihead_softmax_attention, 
    #                                     in_axes=(0, None, None, None, None))
        
    #     # Weigths for self-attention layer
    #     key = jrand.PRNGKey(42)
    #     xkey, qkey, kkey, vkey, okey = jrand.split(key, 5)
    #     X = jrand.normal(xkey, (batchsize, embedding_dim, seq_len))
    #     WQ = jrand.normal(qkey, (dk*num_heads, embedding_dim))
    #     WK = jrand.normal(kkey, (dk*num_heads, embedding_dim))
    #     WV = jrand.normal(vkey, (dk*num_heads, embedding_dim))
    #     WO = jrand.normal(okey, (embedding_dim, dk*num_heads))

    #     print(jax.make_jaxpr(vmap_multihead_attn)(X, WQ, WK, WV, WO))

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(vmap_multihead_attn, order="rev", argnums=(1, 2, 3, 4)))
        
    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(vmap_multihead_attn, argnums=(1, 2, 3, 4)))
    #     revres = jax_jac_rev(X, WQ, WK, WV, WO)
    #     veres = jac_rev(X, WQ, WK, WV, WO)
        
    #     print("ve", veres[0].sum())
    #     print("jax", revres[0].sum())
        
    #     print("ve", veres[1].sum())
    #     print("jax", revres[1].sum())
        
    #     print("ve", veres[2].sum())
    #     print("jax", revres[2].sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_two_layer_transformer(self):
    #     def softmax_attention(X, WQ, WK, WV):
    #         q = WQ @ X
    #         k = WK @ X
    #         v = WV @ X
    #         a = q @ k.T
    #         return jnn.softmax(a, axis=0) @ v
                
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

