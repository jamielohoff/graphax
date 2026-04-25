import time
import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose

from _transformer import (make_weights, glorot, silu,
                        make_positional_encoding, softmax_ce_loss,
                        multihead_attention_block, MLP,
                        efficient_multihead_softmax_attention)


def softmax_attention(X, WQ, WK, WV):
    q = X @ WQ  # (dk, seq_len)
    k = X @ WK
    v = X @ WV
    dk = float(q.shape[0])
    attn_scores = q @ k.T / jnp.sqrt(dk)  # (seq_len, seq_len)
    attn_weights = jnn.softmax(attn_scores, axis=-1)
    return attn_weights @ v # (dk, seq_len)

class TransformerTest(unittest.TestCase):
    ### Testing basic softmax
    def test_softmax_0(self):
        print("Testing softmax with summing along axis=0...")
        def f(x, y):
            return jnp.sin(jnn.softmax(x @ y, axis=0))

        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 4))

        print(jax.make_jaxpr(f)(x, y))
        print(jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y))
        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jit(jax.jacrev(f, argnums=(0, 1)))(x, y)

        print("ve", veres[0])
        print("jax", revres[0])

        self.assertTrue(tree_allclose(veres, revres))

    def test_softmax_1(self):
        print("Testing softmax with summing along axis=1...")
        def f(x, y):
            return jnp.sin(jnn.softmax(x @ y, axis=1))
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (2, 3))
        y = jrand.normal(ykey, (3, 4))

        print(jax.make_jaxpr(f)(x, y))
        deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(f, argnums=(0, 1))(x, y)

        print(veres[1])
        print(revres[1])

        self.assertTrue(tree_allclose(veres, revres))

    ### Test of the utility building blocks
    def test_cross_entropy(self):
        print("Testing softmax CE...")
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (32, 10))
        y = jrand.normal(ykey, (32, 10))

        deriv_fn = jax.jit(jacve(softmax_ce_loss, order="rev", argnums=(0, 1)))
        veres = deriv_fn(x, y)

        revres = jax.jacrev(softmax_ce_loss, argnums=(0, 1))(x, y)

        self.assertTrue(tree_allclose(veres, revres))

    ### Test softmax attention
    def test_softmax_self_attention_fwd(self):
        print("Testing softmax self attention with forward-mode AD...")

        key = jrand.PRNGKey(42)
        xkey, qkey, kkey, vkey = jrand.split(key, 4)
        s = 10
        x = glorot(xkey, (16, s))
        WQ = glorot(qkey, (s, 2*s))
        WK = glorot(kkey, (s, 2*s))
        WV = glorot(vkey, (s, 2*s))

        print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

        jax_jac_fwd = jax.jit(jax.jacfwd(softmax_attention, argnums=(1, 2, 3)))
        revres = jax_jac_fwd(x, WQ, WK, WV)

        jac_fwd = jax.jit(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))
        veres = jac_fwd(x, WQ, WK, WV)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"ve {i+1}", ve.sum(), f"jax {i+1}", rev.sum())

        self.assertTrue(tree_allclose(veres, revres))

    def test_softmax_self_attention_rev(self):
        print("Testing softmax self attention with reverse-mode AD...")
        seq_len = 32
        embedding_dim = 64
        dk = 64

        key = jrand.PRNGKey(42)
        xkey, qkey, kkey, vkey = jrand.split(key, 4)
        x = jrand.normal(xkey, (seq_len, embedding_dim))
        WQ = glorot(qkey, (embedding_dim, dk))
        WK = glorot(kkey, (embedding_dim, dk))
        WV = glorot(vkey, (embedding_dim, dk))

        print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))

        jax_jac_rev = jax.jit(jax.jacrev(softmax_attention, argnums=(1, 2, 3)))
        revres = jax_jac_rev(x, WQ, WK, WV)

        jac_rev = jax.jit(jacve(softmax_attention, order="rev", argnums=(1, 2, 3)))
        veres = jac_rev(x, WQ, WK, WV)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).sum())

        self.assertTrue(tree_allclose(veres, revres))

    ### Testing MLP
    def test_MLP(self):
        print("Testing MLP...")
        seq_len = 20
        embedding_dim = 16

        key = jrand.PRNGKey(42)
        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (embedding_dim, 4*embedding_dim))
        b1 = jnp.zeros(4*embedding_dim)
        W2 = glorot(W2key, (4*embedding_dim, embedding_dim))
        b2 = jnp.zeros(embedding_dim)

        x = jrand.normal(key, (seq_len, embedding_dim))

        print(jax.make_jaxpr(MLP)(x, W1, b1, W2, b2))

        deriv_fn = jax.jit(jacve(MLP, order="rev", argnums=(1, 2, 3, 4)))
        veres = deriv_fn(x, W1, b1, W2, b2)

        revres = jax.jacrev(MLP, argnums=(1, 2, 3, 4))(x, W1, b1, W2, b2)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).sum())

        self.assertTrue(tree_allclose(veres, revres))

    ### Testing multi-headed softmax attention
    def test_multihead_attention(self):
        print("Testing multi-head self-attention...")
        batch_size = 3
        num_heads = 4
        seq_len = 10
        embedding_dim = 32
        dk = 32

        key = jrand.PRNGKey(42)
        x = jrand.normal(key, (batch_size, seq_len, embedding_dim))
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (embedding_dim, 3*dk*num_heads))
        WO = glorot(okey, (dk*num_heads, embedding_dim))

        print(jax.make_jaxpr(efficient_multihead_softmax_attention)(x, WQKV, WO))

        jax_jac_rev = jax.jit(jax.jacrev(efficient_multihead_softmax_attention, argnums=(1, 2)))
        revres = jax_jac_rev(x, WQKV, WO)

        jac_rev = jax.jit(jacve(efficient_multihead_softmax_attention, order="rev", argnums=(1, 2)))
        veres = jac_rev(x, WQKV, WO)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).sum())

        self.assertTrue(tree_allclose(veres, revres))

    def test_multihead_attention_block(self):
        print("Testing multi-head attention block...")
        batch_size = 5
        num_heads = 4
        seq_len = 10
        embedding_dim = 16
        head_dim = 12

        key = jrand.PRNGKey(42)
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (embedding_dim, 3*head_dim*num_heads))
        WO = glorot(okey, (head_dim*num_heads, embedding_dim))

        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (embedding_dim, 4*embedding_dim))
        b1 = jnp.zeros(4*embedding_dim)
        W2 = glorot(W2key, (4*embedding_dim, embedding_dim))
        b2 = jnp.zeros(embedding_dim)

        x = jrand.normal(key, (batch_size, seq_len, embedding_dim))
        weights = (WQKV, WO, W1, b1, W2, b2)

        print(jax.make_jaxpr(multihead_attention_block)(x, *weights))

        argnums = range(1, 7)
        deriv_fn = jax.jit(jacve(multihead_attention_block, order="rev", argnums=argnums))
        veres = deriv_fn(x, *weights)

        jax_deriv_fn = jax.jit(jax.jacrev(multihead_attention_block, argnums=argnums))
        revres = jax_deriv_fn(x, *weights)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).mean())

        self.assertTrue(tree_allclose(veres, revres))

    def test_multihead_attention_2_blocks(self):
        batch_size = 3
        num_heads = 4
        seq_len = 10
        embedding_dim = 32
        dk = 32 // num_heads

        def multiple_blocks(x, WQKV1, WO1, W1, b1, W2, b2,
                                WQKV2, WO2, W3, b3, W4, b4):
            x = multihead_attention_block(x, WQKV1, WO1, W1, b1, W2, b2)
            x = multihead_attention_block(x, WQKV2, WO2, W3, b3, W4, b4)
            x = x[:, 0]
            return x.sum()

        key = jrand.PRNGKey(42)
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV1 = glorot(qkvkey, (embedding_dim, 3*dk*num_heads))
        WO1 = glorot(okey, (dk*num_heads, embedding_dim))

        qkvkey, okey, key = jrand.split(key, 3)
        WQKV2 = glorot(qkvkey, (embedding_dim, 3*dk*num_heads))
        WO2 = glorot(okey, (dk*num_heads, embedding_dim))

        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (embedding_dim, 4*embedding_dim))
        b1 = jnp.zeros(4*embedding_dim)
        W2 = glorot(W2key, (4*embedding_dim, embedding_dim))
        b2 = jnp.zeros(embedding_dim)

        W3key, W4key, key = jrand.split(key, 3)
        W3 = glorot(W3key, (embedding_dim, 4*embedding_dim))
        b3 = jnp.zeros(4*embedding_dim)
        W4 = glorot(W4key, (4*embedding_dim, embedding_dim))
        b4 = jnp.zeros(embedding_dim)

        weights = (WQKV1, WO1, W1, b1, W2, b2, WQKV2, WO2, W3, b3, W4, b4)
        x = jrand.normal(key, (batch_size, seq_len, embedding_dim))

        print(jax.make_jaxpr(multiple_blocks)(x, *weights))

        argnums = list(range(1, 13))

        print(jax.make_jaxpr(jacve(multiple_blocks, order="rev", argnums=argnums))(x, *weights))
        deriv_fn = jax.jit(jacve(multiple_blocks, order="rev", argnums=argnums))
        veres = deriv_fn(x, *weights)

        jax_deriv_fn = jax.jit(jax.jacrev(multiple_blocks, argnums=argnums))
        revres = jax_deriv_fn(x, *weights)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).mean())

        out = jax_deriv_fn(x, *weights)
        st = time.time()
        for i in range(50):
            out = jax_deriv_fn(x, *weights)
        print("jax time", time.time() - st)

        out = deriv_fn(x, *weights)
        st = time.time()
        for i in range(50):
            out = deriv_fn(x, *weights)
        print("graphax time", time.time() - st)

        self.assertTrue(tree_allclose(veres, revres))


    ### Testing transformer architecture
    def test_transformer(self):
        batchsize = 5
        s = 1
        num_heads = 8
        seq_len = s*10
        embedding_dim = s*48
        dk = s*32 // num_heads

        positional_encoding = make_positional_encoding(seq_len+1, embedding_dim)

        def multiple_blocks(x, CT, WQKV1, WO1, W1, b1, W2, b2,
                                    WQKV2, WO2, W3, b3, W4, b4,
                                    WQKV3, WO3, W5, b5, W6, b6,
                                    W7, b7, W8, b8):
            x = jnp.concatenate((CT, x), axis=1)
            x = positional_encoding(x)
            x = multihead_attention_block(x, WQKV1, WO1, W1, b1, W2, b2)
            x = multihead_attention_block(x, WQKV2, WO2, W3, b3, W4, b4)
            x = multihead_attention_block(x, WQKV3, WO3, W5, b5, W6, b6)
            x = x[:, 0]
            return silu(x @ W7+ b7) @ W8 + b8

        def transformer(x, labels, *weights):
            out = multiple_blocks(x, *weights)
            return softmax_ce_loss(out, labels).sum()

        key = jrand.PRNGKey(42)
        W5key, W6key, key = jrand.split(key, 3)
        W7 = glorot(W5key, (embedding_dim, 128))
        b7 = jnp.zeros(128)
        W8 = glorot(W6key, (128, 10))
        b8 = jnp.zeros(10)

        CT = jrand.normal(key, (1, 1, embedding_dim))
        CT = jnp.broadcast_to(CT, (batchsize, 1, embedding_dim))

        block_weights = make_weights(key, 3, dk, num_heads, embedding_dim)
        weights = tuple([CT] + block_weights + [W7, b7, W8, b8])

        x = jrand.normal(key, (batchsize, seq_len, embedding_dim))
        labels = jrand.normal(key, (batchsize, 10))

        jaxpr = jax.make_jaxpr(transformer)(x, labels, *weights)

        argnums = list(range(1, len(weights) + 2))

        jacve_jaxpr = jax.make_jaxpr(jacve(transformer, order="rev", argnums=argnums))(x, labels, *weights)
        print('Cost analysis:')
        print(len([eqn.primitive for eqn in jaxpr.eqns if eqn.primitive is jax.lax.dot_general_p]))
        print(len([eqn.primitive for eqn in jaxpr.eqns if eqn.primitive is jax.lax.mul_p]))
        deriv_fn = jax.jit(jacve(transformer, order="rev", argnums=argnums))
        costs = deriv_fn.lower(x, labels, *weights).compile().cost_analysis()
        print('Tflops', costs['flops'] / 1e12)
        print('GB', sum(val for key, val in costs.items() if "bytes accessed" in key) / 1e9)
        veres = deriv_fn(x, labels, *weights)

        jax_jaxpr = jax.make_jaxpr(jax.jacrev(transformer, argnums=argnums))(x, labels, *weights)
        print('Cost analysis')
        print(len([eqn.primitive for eqn in jax_jaxpr.eqns if eqn.primitive is jax.lax.dot_general_p]))
        print(len([eqn.primitive for eqn in jaxpr.eqns if eqn.primitive is jax.lax.mul_p]))
        jax_deriv_fn = jax.jit(jax.jacrev(transformer, argnums=argnums))
        costs = jax_deriv_fn.lower(x, labels, *weights).compile().cost_analysis()
        print('Tflops', costs['flops'] / 1e12)
        print('GB', sum(val for key, val in costs.items() if "bytes accessed" in key) / 1e9)
        revres = jax_deriv_fn(x, labels, *weights)

        for i, (ve, rev) in enumerate(zip(veres, revres)):
            print(f"err{i+1}", jnp.abs(ve - rev).mean())

        st = time.time()
        for i in range(10):
            out = jax_deriv_fn(x, labels, *weights)
            jax.block_until_ready(out)
        print("jax time", time.time() - st)

        st = time.time()
        for i in range(10):
            out = deriv_fn(x, labels, *weights)
            jax.block_until_ready(out)
        print("graphax time", time.time() - st)

        self.assertTrue(tree_allclose(veres, revres))


if __name__ == '__main__':
    unittest.main()
