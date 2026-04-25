from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

import numpy as np


### Positional encoding
def make_positional_encoding(seq_len, embedding_dim):
    n = 10000
    pe = np.zeros((seq_len, embedding_dim))
    position = np.arange(0, seq_len, dtype=np.float32)[:, jnp.newaxis]
    div_term = np.power(n, jnp.arange(0, embedding_dim, 2) / embedding_dim)[jnp.newaxis, :]
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = jnp.array(pe)

    def positional_encoding(xs):
        return xs + pe[jnp.newaxis, :, :]
    return positional_encoding


### Softmax cross-entropy loss
def _log_softmax(logits, axis=-1):
    return logits - jnp.log(jnp.sum(jnp.exp(logits), axis=axis))


def softmax_ce_loss(logits, labels):
    return -jnp.sum(labels*_log_softmax(logits, axis=0))


### Glorot initialization
def glorot(key, shape):
    return jrand.normal(key, shape)*jnp.sqrt(2/(shape[0] + shape[1]))


### SiLU activation function
def silu(x):
    sig = lax.logistic(x)
    return x * sig


### Multi-head self-attention
def softmax_attn(q, k , v):
    a = q @ k.T / jnp.sqrt(k.shape[1])
    return jnn.softmax(a, axis=-1) @ v


def efficient_multihead_softmax_attention(X, WQKV, WO, num_heads: int = 4):
    batch_size, seq_len, _ = X.shape
    head_dim = WQKV.shape[-1] // (3*num_heads)
    qkv = X @ WQKV
    qkv = qkv.reshape(batch_size, seq_len, num_heads, 3*head_dim)
    q, k, v = jnp.split(qkv, (head_dim, 2*head_dim), axis=-1)

    attn_scores = jnp.einsum('bthd,blhd->bhtl', q, k) / jnp.sqrt(k.shape[1])
    attn_weights = jnn.softmax(attn_scores, axis=-1)

    out = jnp.einsum('bhtl,blhd->bthd', attn_weights, v)
    out = out.reshape(batch_size, seq_len, -1)

    return out @ WO


### MLP implementation
def _project(W, b, X):
    return X @ W + b

def MLP(X, W1, b1, W2, b2):
    out = _project(W1, b1, X)
    out = silu(out)
    return _project(W2, b2, out)


### Root mean square normalization
def rms_norm(x, axis=-1):
    return x / jnp.sqrt(jnp.sum(x**2, axis=axis, keepdims=True))


### Attention Block
def multihead_attention_block(X, WQKV, WO, W1, b1, W2, b2):
    out = rms_norm(X)
    out = efficient_multihead_softmax_attention(out, WQKV, WO)
    out = out + X
    out = out + MLP(rms_norm(out), W1, b1, W2, b2)
    return out


# Generate weights for attention blocks and MLP layers
def make_weights(key, num_attn_blocks: int = 2, dk: int = 512, num_heads: int = 8, embedding_dim: int = 512):
    weights = []
    for _ in range(num_attn_blocks):
        # Weigths for self-attention
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (embedding_dim, dk*num_heads*3))
        WO = glorot(okey, (dk*num_heads, embedding_dim))
        
        # Weights for MLP layers
        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (embedding_dim, 4*embedding_dim))
        b1 = jnp.zeros(4*embedding_dim)
        W2 = glorot(W2key, (4*embedding_dim, embedding_dim))
        b2 = jnp.zeros(embedding_dim)

        weights.extend([WQKV, WO, W1, b1, W2, b2])
    return weights

