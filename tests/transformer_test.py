import time
import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose

from _transformer import (make_weights, glorot, 
                        make_positional_encoding, softmax_ce_loss, gelu,
                        multihead_attention_block)


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
    #     # TODO investigate errors between gradients computed by vertex elimination
    #     # and errors computed through jax
    #     num_heads = 8
    #     seq_len = 16
    #     embedding_dim = 16
    #     dk = 32//num_heads

    #     # Weights for self-attention layer
    #     key = jrand.PRNGKey(42)
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
    #     weights = (WQ, WK, WV, WO, W1, b1, W2, b2)
    #     print(jax.make_jaxpr(multihead_attention_block)(x, *weights))
        
    #     argnums = range(1, 9)
    #     print(jax.make_jaxpr(jacve(multihead_attention_block, order="rev", argnums=argnums))(x, *weights))
    #     deriv_fn = jax.jit(jacve(multihead_attention_block, order="rev", argnums=argnums))
    #     veres = deriv_fn(x, *weights)

    #     jax_deriv_fn = jax.jit(jax.jacrev(multihead_attention_block, argnums=argnums))
    #     revres = jax_deriv_fn(x, *weights)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).mean())
    #     print("err2", jnp.abs(veres[1] - revres[1]).mean())
    #     print("err3", jnp.abs(veres[2] - revres[2]).mean())
    #     print("err4", jnp.abs(veres[3] - revres[3]).mean())
        
    #     print("err5", jnp.abs(veres[4] - revres[4]).mean())
    #     print("err6", jnp.abs(veres[5] - revres[5]).mean())
    #     print("err7", jnp.abs(veres[6] - revres[6]).mean())
    #     print("err8", jnp.abs(veres[7] - revres[7]).mean())
        
    #     import matplotlib.pyplot as plt
    #     import time
        
    #     out = deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     st = time.time()
    #     for i in range(50):
    #         out = deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     print("graphax time", time.time() - st)
        
    #     jax_deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     st = time.time()
    #     for i in range(50):
    #         out = jax_deriv_fn(x, WQ, WK, WV, WO, W1, b1, W2, b2)
    #     print("jax time", time.time() - st)
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_multihead_attention_2_blocks(self):
    #     num_heads = 8
    #     seq_len = 128
    #     embedding_dim = 256
    #     dk = 256//num_heads
        
    #     def multiple_blocks(x, WQ1, WK1, WV1, WO1, W1, b1, W2, b2,
    #                         WQ2, WK2, WV2, WO2, W3, b3, W4, b4):
    #         x = multihead_attention_block(x, WQ1, WK1, WV1, WO1, W1, b1, W2, b2)
    #         x = multihead_attention_block(x, WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
    #         x = x[:, 0]
    #         return x.sum()

    #     # Weights for self-attention layer
    #     key = jrand.PRNGKey(42)
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ1 = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK1 = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV1 = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO1 = glorot(okey, (embedding_dim, dk*num_heads))
        
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ2 = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK2 = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV2 = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO2 = glorot(okey, (embedding_dim, dk*num_heads))
        
    #     # Weights for MLP layer
    #     W1key, W2key, key = jrand.split(key, 3)
    #     W1 = glorot(W1key, (1024, embedding_dim))
    #     b1 = jnp.zeros((1024,), dtype=jnp.float32)
    #     W2 = glorot(W2key, (embedding_dim, 1024))
    #     b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)
        
    #     W3key, W4key, key = jrand.split(key, 3)
    #     W3 = glorot(W3key, (1024, embedding_dim))
    #     b3 = jnp.zeros((1024,), dtype=jnp.float32)
    #     W4 = glorot(W4key, (embedding_dim, 1024))
    #     b4 = jnp.zeros((embedding_dim,), dtype=jnp.float32)
        
    #     weights = (WQ1, WK1, WV1, WO1, W1, b1, W2, b2,
    #                 WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
        
    #     x = jrand.normal(key, (embedding_dim, seq_len))
        
    #     print(jax.make_jaxpr(multiple_blocks)(x, *weights))
        
    #     argnums = list(range(1, 17))
        
    #     print(jax.make_jaxpr(jacve(multiple_blocks, order="rev", argnums=argnums))(x, *weights))
    #     deriv_fn = jax.jit(jacve(multiple_blocks, order="rev", argnums=argnums))
    #     veres = deriv_fn(x, *weights)

    #     jax_deriv_fn = jax.jit(jax.jacrev(multiple_blocks, argnums=argnums))
    #     revres = jax_deriv_fn(x, *weights)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).mean())
    #     print("err2", jnp.abs(veres[1] - revres[1]).mean())
    #     print("err3", jnp.abs(veres[2] - revres[2]).mean())
    #     print("err4", jnp.abs(veres[3] - revres[3]).mean())
        
    #     print("err5", jnp.abs(veres[4] - revres[4]).mean())
    #     print("err6", jnp.abs(veres[5] - revres[5]).mean())
    #     print("err7", jnp.abs(veres[6] - revres[6]).mean())
    #     print("err8", jnp.abs(veres[7] - revres[7]).mean())
        
    #     print("err9", jnp.abs(veres[8] - revres[8]).mean())
    #     print("err10", jnp.abs(veres[9] - revres[9]).mean())
    #     print("err11", jnp.abs(veres[10] - revres[10]).mean())
    #     print("err12", jnp.abs(veres[11] - revres[11]).mean())
        
    #     print("err13", jnp.abs(veres[12] - revres[12]).mean())
    #     print("err14", jnp.abs(veres[13] - revres[13]).mean())
    #     print("err15", jnp.abs(veres[14] - revres[14]).mean())
    #     print("err16", jnp.abs(veres[15] - revres[15]).mean())
        
    #     import matplotlib.pyplot as plt
    #     import time
        
    #     out = jax_deriv_fn(x, *weights)
    #     st = time.time()
    #     for i in range(50):
    #         out = jax_deriv_fn(x, *weights)
    #     print("jax time", time.time() - st)
        
    #     out = deriv_fn(x, *weights)
    #     st = time.time()
    #     for i in range(50):
    #         out = deriv_fn(x, *weights)
    #     print("graphax time", time.time() - st)
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
    # def test_vmap_multihead_attention_2_blocks(self):
    #     # TODO investigate errors between gradients computed by vertex elimination
    #     # and errors computed through jax
    #     batchsize = 4
    #     s = 1
    #     num_heads = 8
    #     seq_len = s*16
    #     embedding_dim = s*192
    #     dk = s*256//num_heads
        
    #     @partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, None,
    #                                     None, None, None, None, None, None, None, None))
    #     def multiple_blocks(x, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2,
    #                         WQ2, WK2, WV2, WO2, W3, b3, W4, b4):
    #         x = jnp.concatenate((CT, x), axis=1)
    #         x = multihead_attention_block(x, WQ1, WK1, WV1, WO1, W1, b1, W2, b2)
    #         x = multihead_attention_block(x, WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
    #         return x[:, 0]
        
    #     def transformer(x, *weights):
    #         return multiple_blocks(x, *weights).sum()

    #     # Weights for self-attention layers
    #     key = jrand.PRNGKey(42)
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ1 = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK1 = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV1 = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO1 = glorot(okey, (embedding_dim, dk*num_heads))
        
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ2 = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK2 = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV2 = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO2 = glorot(okey, (embedding_dim, dk*num_heads))
        
    #     # Weights for MLP layers
    #     W1key, W2key, key = jrand.split(key, 3)
    #     W1 = glorot(W1key, (512, embedding_dim))
    #     b1 = jnp.zeros((512,), dtype=jnp.float32)
    #     W2 = glorot(W2key, (embedding_dim, 512))
    #     b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)
        
    #     W3key, W4key, key = jrand.split(key, 3)
    #     W3 = glorot(W3key, (512, embedding_dim))
    #     b3 = jnp.zeros((512,), dtype=jnp.float32)
    #     W4 = glorot(W4key, (embedding_dim, 512))
    #     b4 = jnp.zeros((embedding_dim,), dtype=jnp.float32)
        
    #     CT = jrand.normal(key, (embedding_dim, 1))
        
    #     weights = (CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2,
    #                     WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
        
    #     x = jrand.normal(key, (batchsize, embedding_dim, seq_len))
        
    #     print(jax.make_jaxpr(transformer)(x, *weights))
        
    #     argnums = list(range(1, 18))
        
    #     # print(jax.make_jaxpr(jacve(transformer, order="rev", argnums=argnums))(x, *weights))
    #     deriv_fn = jax.jit(jacve(transformer, order="rev", argnums=argnums))
    #     veres = deriv_fn(x, *weights)

    #     # print(jax.make_jaxpr(jax.jacrev(transformer, argnums=argnums))(x, *weights))
    #     jax_deriv_fn = jax.jit(jax.jacrev(transformer, argnums=argnums))
    #     revres = jax_deriv_fn(x, *weights)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).mean())
        
    #     print("err2", jnp.abs(veres[1] - revres[1]).mean())
    #     print("err3", jnp.abs(veres[2] - revres[2]).mean())
    #     print("err4", jnp.abs(veres[3] - revres[3]).mean())
    #     print("err5", jnp.abs(veres[4] - revres[4]).mean())
        
    #     print("err6", jnp.abs(veres[5] - revres[5]).mean())
    #     print("err7", jnp.abs(veres[6] - revres[6]).mean())
    #     print("err8", jnp.abs(veres[7] - revres[7]).mean())
    #     print("err9", jnp.abs(veres[8] - revres[8]).mean())
        
    #     print("err10", jnp.abs(veres[9] - revres[9]).mean())
    #     print("err11", jnp.abs(veres[10] - revres[10]).mean())
    #     print("err12", jnp.abs(veres[11] - revres[11]).mean())
    #     print("err13", jnp.abs(veres[12] - revres[12]).mean())
        
    #     print("err14", jnp.abs(veres[13] - revres[13]).mean())
    #     print("err15", jnp.abs(veres[14] - revres[14]).mean())
    #     print("err16", jnp.abs(veres[15] - revres[15]).mean())
    #     print("err16", jnp.abs(veres[16] - revres[16]).mean())
                
    #     import time
        
    #     out = jax_deriv_fn(x, *weights)
    #     st = time.time()
    #     for i in range(50):
    #         out = jax_deriv_fn(x, *weights)
    #     print("jax time", time.time() - st)
        
    #     out = deriv_fn(x, *weights)
    #     st = time.time()
    #     for i in range(50):
    #         out = deriv_fn(x, *weights)
    #     print("graphax time", time.time() - st)
        
    #     self.assertTrue(tree_allclose(veres, revres))
            
    # ### Test all the softmax attention crap
    # def test_softmax_0(self):
    #     def f(x, y):
    #         return jnp.sin(jnn.softmax(x @ y, axis=0))
        
    #     key = jrand.PRNGKey(42)
    #     xkey, ykey = jrand.split(key, 2)
    #     x = jrand.normal(xkey, (2, 3))
    #     y = jrand.normal(ykey, (3, 4))

    #     print(jax.make_jaxpr(f)(x, y))
    #     print(jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y))
    #     deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
    #     veres = deriv_fn(x, y)

    #     # print(jax.make_jaxpr(jax.jacrev(f, argnums=(0, 1)))(x, y))
    #     revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)
        
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
    
    # def test_multihead_attention(self):
    #     num_heads = 1
    #     seq_len = 32
    #     embedding_dim = 32
    #     dk = 32//num_heads
        
    #     key = jrand.PRNGKey(42)
    #     x = jrand.normal(key, (embedding_dim, seq_len))
    #     qkey, kkey, vkey, okey, key = jrand.split(key, 5)
    #     WQ = glorot(qkey, (dk*num_heads, embedding_dim))
    #     WK = glorot(kkey, (dk*num_heads, embedding_dim))
    #     WV = glorot(vkey, (dk*num_heads, embedding_dim))
    #     WO = glorot(okey, (embedding_dim, dk*num_heads))

    #     print(jax.make_jaxpr(multihead_softmax_attention)(x, WQ, WK, WV, WO))

    #     # print("jax jaxpr", jax.make_jaxpr(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jax_jac_rev = jax.jit(jax.jacrev(multihead_softmax_attention, argnums=(1, 2, 3, 4)))
    #     revres = jax_jac_rev(x, WQ, WK, WV, WO)

    #     # print("ve jaxpr", jax.make_jaxpr(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))(x, WQ, WK, WV))
    #     jac_rev = jax.jit(jacve(multihead_softmax_attention, order="rev", argnums=(1, 2, 3, 4)))
    #     veres = jac_rev(x, WQ, WK, WV, WO)
        
    #     print("err1", jnp.abs(veres[0] - revres[0]).sum())
    #     print("err2", jnp.abs(veres[1] - revres[1]).sum())
    #     print("err3", jnp.abs(veres[2] - revres[2]).sum())
    #     print("err4", jnp.abs(veres[3] - revres[3]).sum())
        
    #     self.assertTrue(tree_allclose(veres, revres))
        
    def test_vmap_transformer(self):
        batchsize = 8
        s = 1
        num_heads = 8
        seq_len = s*16
        embedding_dim = s*192
        dk = s*256//num_heads
        
        positional_encoding = make_positional_encoding(seq_len+1, embedding_dim)
        
        @partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, None,
                                        None, None, None, None, None, None, None, None, None,
                                        None, None, None, None, None, None, None, None, None,
                                        None, None))
        def multiple_blocks(x, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2,
                                    WQ2, WK2, WV2, WO2, W3, b3, W4, b4, 
                                    WQ3, WK3, WV3, WO3, W5, b5, W6, b6, 
                                    W7, b7, W8, b8):
            x = jnp.concatenate((CT, x), axis=1)
            x = positional_encoding(x)
            
            x = multihead_attention_block(x, WQ1, WK1, WV1, WO1, W1, b1, W2, b2)
            x = multihead_attention_block(x, WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
            x = multihead_attention_block(x, WQ3, WK3, WV3, WO3, W5, b5, W6, b6)
            x = x[:, 0]
            return W8 @ gelu(W7 @ x + b7) + b8
        
        def transformer(x, labels, *weights):
            out = multiple_blocks(x, *weights)
            return softmax_ce_loss(out, labels).sum()

        # Weights for self-attention layers
        key = jrand.PRNGKey(42)        
        W5key, W6key, key = jrand.split(key, 3)
        W7 = glorot(W5key, (256, embedding_dim))
        b7 = jnp.zeros(256, dtype=jnp.float32)
        W8 = glorot(W6key, (10, 256))
        b8 = jnp.zeros(10, dtype=jnp.float32)
        
        CT = jrand.normal(key, (embedding_dim, 1))
        
        weights = make_weights(key, 3, dk, num_heads, embedding_dim)
        weights = [CT] + weights + [W7, b7, W8, b8]
        weights = tuple(weights)
        
        x = jrand.normal(key, (batchsize, embedding_dim, seq_len))
        labels = jrand.normal(key, (batchsize, 10))
        
        jaxpr = jax.make_jaxpr(transformer)(x, labels, *weights)
        
        argnums = list(range(2, len(weights) + 2))
        
        jacve_jaxpr = jax.make_jaxpr(jacve(transformer, order="rev", argnums=argnums))(x, labels, *weights)
        deriv_fn = jax.jit(jacve(transformer, order="rev", argnums=argnums))
        veres = deriv_fn(x, labels, *weights)

        jax_jaxpr = jax.make_jaxpr(jax.jacrev(transformer, argnums=argnums))(x, labels, *weights)
        jax_deriv_fn = jax.jit(jax.jacrev(transformer, argnums=argnums))
        revres = jax_deriv_fn(x, labels, *weights)
        
        print("err1", jnp.abs(veres[0] - revres[0]).mean())
        
        print("err2", jnp.abs(veres[1] - revres[1]).mean())
        print("err3", jnp.abs(veres[2] - revres[2]).mean())
        print("err4", jnp.abs(veres[3] - revres[3]).mean())
        print("err5", jnp.abs(veres[4] - revres[4]).mean())
        
        print("err6", jnp.abs(veres[5] - revres[5]).mean())
        print("err7", jnp.abs(veres[6] - revres[6]).mean())
        print("err8", jnp.abs(veres[7] - revres[7]).mean())
        print("err9", jnp.abs(veres[8] - revres[8]).mean())
        
        print("err10", jnp.abs(veres[9] - revres[9]).mean())
        print("err11", jnp.abs(veres[10] - revres[10]).mean())
        print("err12", jnp.abs(veres[11] - revres[11]).mean())
        print("err13", jnp.abs(veres[12] - revres[12]).mean())
        
        print("err14", jnp.abs(veres[13] - revres[13]).mean())
        print("err15", jnp.abs(veres[14] - revres[14]).mean())
        print("err16", jnp.abs(veres[15] - revres[15]).mean())
        print("err17", jnp.abs(veres[16] - revres[16]).mean())
                
        st = time.time()
        for i in range(50):
            out = jax_deriv_fn(x, labels, *weights)
        print("jax time", time.time() - st)
        
        st = time.time()
        for i in range(50):
            out = deriv_fn(x, labels, *weights)
        print("graphax time", time.time() - st)
        
        from graphax.sparse.utils import count_muls
        
        num_muls = sum([count_muls(p) for p in jaxpr.jaxpr.eqns])
        num_dots_jacve = sum([count_muls(p) for p in jacve_jaxpr.jaxpr.eqns])
        num_dots_jax = sum([count_muls(p) for p in jax_jaxpr.jaxpr.eqns])
        
        print(num_dots_jacve - num_muls, num_dots_jax - num_muls)
        
        self.assertTrue(tree_allclose(veres, revres))
        
        


if __name__ == '__main__':
    unittest.main()

