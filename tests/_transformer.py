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
    pe = np.zeros((embedding_dim, seq_len))
    position = np.arange(0, seq_len, dtype=np.float32)[None, :]
    div_term = np.power(n, jnp.arange(0, embedding_dim, 2) / embedding_dim)[:, None]
    pe[0::2, :] = np.sin(position * div_term)
    pe[1::2, :] = np.cos(position * div_term)
    pe = jnp.array(pe)

    def positional_encoding(xs):
        return xs + pe[:xs.shape[0], :]
    return positional_encoding


### Softmax cross-entropy loss
def _log_softmax(logits, axis=0):
    return logits - jnp.log(jnp.sum(jnp.exp(logits), axis=axis))

@jax.vmap
def softmax_ce_loss(logits, labels):
    return -jnp.sum(labels*_log_softmax(logits, axis=0))


### Glorot initialization
def glorot(key, shape):
    return jrand.normal(key, shape)*jnp.sqrt(2/(shape[0] + shape[1]))


### GeLU activation function
def gelu(x):
    return x/2*(1 + lax.erf(x/jnp.sqrt(2.0)))


### Multi-head self-attention
def softmax_attn(q, k, v):
    a = q @ k.T / jnp.sqrt(k.shape[0])
    return jnn.softmax(a, axis=0) @ v

def _proj_head(W, X, num_heads: int = 8):
    return jnp.reshape(W @ X, (X.shape[-1], num_heads, -1)) 

def efficient_multihead_softmax_attention(X, WQKV, WO, num_heads: int = 8):
    qkv = _proj_head(WQKV, X, num_heads=num_heads)
    q, k, v = jnp.split(qkv, 3, axis=-1)
    out = jax.vmap(softmax_attn, in_axes=(1, 1, 1), out_axes=1)(q, k, v)
    out = jnp.reshape(out, (-1, X.shape[-1]))
    return WO @ out 


### MLP implementation
@partial(jax.vmap, in_axes=(None, None, 1), out_axes=1)
def _project(W, b, X):
    return W @ X + b

def MLP(X, W1, b1, W2, b2):
    out = _project(W1, b1, X)
    out = gelu(out)
    return _project(W2, b2, out)


### Layer normalization
def variance(X, axis=0):
    return jnp.mean(jnp.square(X - jnp.mean(X, axis=axis)), axis=axis)

@jax.vmap
def layer_norm(X):  
    mean = jnp.mean(X, axis=0)
    var = variance(X, axis=0)
    return (X - mean)/jnp.sqrt(var + 1e-7)


### Attention Block
def multihead_attention_block(X, WQKV, WO, W1, b1, W2, b2):
    out = layer_norm(X)
    out = efficient_multihead_softmax_attention(out, WQKV, WO)
    out = out + X
    out = layer_norm(out)
    out = MLP(out, W1, b1, W2, b2)
    return out + X


# Generate weights for attention blocks and MLP layers
def make_weights(key, num_attn_blocks: int = 2, dk: int = 512, num_heads: int = 8, embedding_dim: int = 512):
    weights = []
    for _ in range(num_attn_blocks):
        # Weigths for self-attention
        qkvkey, okey, key = jrand.split(key, 3)
        WQKV = glorot(qkvkey, (dk*num_heads*3, embedding_dim))
        WO = glorot(okey, (embedding_dim, dk*num_heads))
        
        # Weights for MLP layers
        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (1024, embedding_dim))
        b1 = jnp.zeros((1024,))
        W2 = glorot(W2key, (embedding_dim, 1024))
        b2 = jnp.zeros((embedding_dim,))

        weights.extend([WQKV, WO, W1, b1, W2, b2])
    return weights

