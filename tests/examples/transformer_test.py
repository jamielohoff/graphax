import time
import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose

from _transformer import (make_weights, glorot,
                        make_positional_encoding, softmax_ce_loss, gelu,
                        multihead_attention_block, MLP,
                        efficient_multihead_softmax_attention)


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
        def softmax_attention(X, WQ, WK, WV):
            q = WQ @ X  # (dk, seq_len)
            k = WK @ X
            v = WV @ X
            dk = float(q.shape[0])
            a = q.T @ k / jnp.sqrt(dk)  # (seq_len, seq_len)
            return v @ jnn.softmax(a, axis=-1)  # (dk, seq_len)

        key = jrand.PRNGKey(42)
        xkey, qkey, kkey, vkey = jrand.split(key, 4)
        s = 10
        x = glorot(xkey, (s, 2*s))
        WQ = glorot(qkey, (s, s))
        WK = glorot(kkey, (s, s))
        WV = glorot(vkey, (s, s))

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

        def softmax_attention(X, WQ, WK, WV):
            q = WQ @ X  # (dk, seq_len)
            k = WK @ X
            v = WV @ X
            a = q.T @ k / jnp.sqrt(float(dk))  # (seq_len, seq_len)
            return v @ jnn.softmax(a, axis=-1)  # (dk, seq_len)

        key = jrand.PRNGKey(42)
        xkey, qkey, kkey, vkey = jrand.split(key, 4)
        x = jrand.normal(xkey, (embedding_dim, seq_len))
        WQ = glorot(qkey, (dk, embedding_dim))
        WK = glorot(kkey, (dk, embedding_dim))
        WV = glorot(vkey, (dk, embedding_dim))

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
        embedding_dim = 15

        key = jrand.PRNGKey(42)
        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (10, embedding_dim))
        b1 = jnp.zeros((10,), dtype=jnp.float32)
        W2 = glorot(W2key, (embedding_dim, 10))
        b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        x = jrand.normal(key, (embedding_dim, seq_len))

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
        num_heads = 1
        seq_len = 32
        embedding_dim = 32
        dk = 32 // num_heads

        key = jrand.PRNGKey(42)
        x = jrand.normal(key, (embedding_dim, seq_len))
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO = glorot(okey, (embedding_dim, dk*num_heads))

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
        num_heads = 4
        seq_len = 16
        embedding_dim = 16
        dk = 32 // num_heads

        key = jrand.PRNGKey(42)
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO = glorot(okey, (embedding_dim, dk*num_heads))

        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (64, embedding_dim))
        b1 = jnp.zeros((64,), dtype=jnp.float32)
        W2 = glorot(W2key, (embedding_dim, 64))
        b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        x = jrand.normal(key, (embedding_dim, seq_len))
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
        WQKV1 = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO1 = glorot(okey, (embedding_dim, dk*num_heads))

        qkvkey, okey, key = jrand.split(key, 3)
        WQKV2 = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO2 = glorot(okey, (embedding_dim, dk*num_heads))

        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (64, embedding_dim))
        b1 = jnp.zeros((64,), dtype=jnp.float32)
        W2 = glorot(W2key, (embedding_dim, 64))
        b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        W3key, W4key, key = jrand.split(key, 3)
        W3 = glorot(W3key, (64, embedding_dim))
        b3 = jnp.zeros((64,), dtype=jnp.float32)
        W4 = glorot(W4key, (embedding_dim, 64))
        b4 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        weights = (WQKV1, WO1, W1, b1, W2, b2, WQKV2, WO2, W3, b3, W4, b4)
        x = jrand.normal(key, (embedding_dim, seq_len))

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

    def test_vmap_multihead_attention_2_blocks(self):
        batchsize = 4
        s = 1
        num_heads = 4
        seq_len = s*16
        embedding_dim = s*48
        dk = s*32 // num_heads

        @partial(jax.vmap, in_axes=(0,) + (None,) * 13)
        def multiple_blocks(x, CT, WQKV1, WO1, W1, b1, W2, b2,
                            WQKV2, WO2, W3, b3, W4, b4):
            x = jnp.concatenate((CT, x), axis=1)
            x = multihead_attention_block(x, WQKV1, WO1, W1, b1, W2, b2)
            x = multihead_attention_block(x, WQKV2, WO2, W3, b3, W4, b4)
            return x[:, 0]

        def transformer(x, *weights):
            return multiple_blocks(x, *weights).sum()

        key = jrand.PRNGKey(42)
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV1 = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO1 = glorot(okey, (embedding_dim, dk*num_heads))

        qkvkey, okey, key = jrand.split(key, 3)
        WQKV2 = glorot(qkvkey, (3*dk*num_heads, embedding_dim))
        WO2 = glorot(okey, (embedding_dim, dk*num_heads))

        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (512, embedding_dim))
        b1 = jnp.zeros((512,), dtype=jnp.float32)
        W2 = glorot(W2key, (embedding_dim, 512))
        b2 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        W3key, W4key, key = jrand.split(key, 3)
        W3 = glorot(W3key, (512, embedding_dim))
        b3 = jnp.zeros((512,), dtype=jnp.float32)
        W4 = glorot(W4key, (embedding_dim, 512))
        b4 = jnp.zeros((embedding_dim,), dtype=jnp.float32)

        CT = jrand.normal(key, (embedding_dim, 1))

        weights = (CT, WQKV1, WO1, W1, b1, W2, b2, WQKV2, WO2, W3, b3, W4, b4)
        x = jrand.normal(key, (batchsize, embedding_dim, seq_len))

        print(jax.make_jaxpr(transformer)(x, *weights))

        argnums = list(range(1, len(weights) + 1))

        deriv_fn = jax.jit(jacve(transformer, order="rev", argnums=argnums))
        veres = deriv_fn(x, *weights)

        print(jax.make_jaxpr(deriv_fn)(x, *weights))

        jax_deriv_fn = jax.jit(jax.jacrev(transformer, argnums=argnums))
        revres = jax_deriv_fn(x, *weights)

        print(jax.make_jaxpr(jax_deriv_fn)(x, *weights))

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
    # def test_vmap_transformer(self):
    #     batchsize = 8
    #     s = 1
    #     num_heads = 8
    #     seq_len = s*16
    #     embedding_dim = s*48
    #     dk = s*32 // num_heads

    #     positional_encoding = make_positional_encoding(seq_len+1, embedding_dim)

    #     @partial(jax.vmap, in_axes=(0,) + (None,) * 23)
    #     def multiple_blocks(x, CT, WQKV1, WO1, W1, b1, W2, b2,
    #                                 WQKV2, WO2, W3, b3, W4, b4,
    #                                 WQKV3, WO3, W5, b5, W6, b6,
    #                                 W7, b7, W8, b8):
    #         x = jnp.concatenate((CT, x), axis=1)
    #         x = positional_encoding(x)
    #         x = multihead_attention_block(x, WQKV1, WO1, W1, b1, W2, b2)
    #         x = multihead_attention_block(x, WQKV2, WO2, W3, b3, W4, b4)
    #         x = multihead_attention_block(x, WQKV3, WO3, W5, b5, W6, b6)
    #         x = x[:, 0]
    #         return W8 @ gelu(W7 @ x + b7) + b8

    #     def transformer(x, labels, *weights):
    #         out = multiple_blocks(x, *weights)
    #         return softmax_ce_loss(out, labels).sum()

    #     key = jrand.PRNGKey(42)
    #     W5key, W6key, key = jrand.split(key, 3)
    #     W7 = glorot(W5key, (32, embedding_dim))
    #     b7 = jnp.zeros(32, dtype=jnp.float32)
    #     W8 = glorot(W6key, (10, 32))
    #     b8 = jnp.zeros(10, dtype=jnp.float32)

    #     CT = jrand.normal(key, (embedding_dim, 1))

    #     block_weights = make_weights(key, 3, dk, num_heads, embedding_dim)
    #     weights = tuple([CT] + block_weights + [W7, b7, W8, b8])

    #     x = jrand.normal(key, (batchsize, embedding_dim, seq_len))
    #     labels = jrand.normal(key, (batchsize, 10))

    #     jaxpr = jax.make_jaxpr(transformer)(x, labels, *weights)

    #     argnums = list(range(2, len(weights) + 2))

    #     jacve_jaxpr = jax.make_jaxpr(jacve(transformer, order="rev", argnums=argnums))(x, labels, *weights)
    #     deriv_fn = jax.jit(jacve(transformer, order="rev", argnums=argnums))
    #     veres = deriv_fn(x, labels, *weights)

    #     jax_jaxpr = jax.make_jaxpr(jax.jacrev(transformer, argnums=argnums))(x, labels, *weights)
    #     jax_deriv_fn = jax.jit(jax.jacrev(transformer, argnums=argnums))
    #     revres = jax_deriv_fn(x, labels, *weights)

    #     for i, (ve, rev) in enumerate(zip(veres, revres)):
    #         print(f"err{i+1}", jnp.abs(ve - rev).mean())

    #     st = time.time()
    #     for i in range(50):
    #         out = jax_deriv_fn(x, labels, *weights)
    #     print("jax time", time.time() - st)

    #     st = time.time()
    #     for i in range(50):
    #         out = deriv_fn(x, labels, *weights)
    #     print("graphax time", time.time() - st)

    #     from graphax.sparse.utils import count_muls

    #     num_muls = sum([count_muls(p) for p in jaxpr.jaxpr.eqns])
    #     num_dots_jacve = sum([count_muls(p) for p in jacve_jaxpr.jaxpr.eqns])
    #     num_dots_jax = sum([count_muls(p) for p in jax_jaxpr.jaxpr.eqns])

    #     print("graphax muls", num_dots_jacve - num_muls, "jax muls", num_dots_jax - num_muls)

    #     self.assertTrue(tree_allclose(veres, revres))


if __name__ == '__main__':
    unittest.main()
