from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

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
@jax.vmap
def softmax_ce_loss(logits, labels):
    return -jnp.sum(labels*jnn.log_softmax(logits, axis=0))


### Glorot initialization
def glorot(key, shape):
    return jrand.normal(key, shape)*jnp.sqrt(2/(shape[0] + shape[1]))


### GELU activation function
def gelu(x):
    return x/2*(1 + lax.erf(x/jnp.sqrt(2)))


### Multi-head self-attention
def multihead_softmax_attention(X, WQ, WK, WV, WO):
    q = WQ @ X
    k = WK @ X
    v = WV @ X
    a = q @ k.T
    out = jnn.softmax(a, axis=0) @ v
    return WO @ out 


### MLP implementation
def _project(W, X):
    return jax.vmap(jnp.matmul, in_axes=(0, None))(W, X)

def MLP(X, W1, b1, W2, b2):
    out = _project(W1, X)
    out = gelu(out + b1)
    return _project(W2, out) + b2


### Softmax cross-entropy loss
@jax.vmap
def softmax_ce_loss(logits, labels):
    return -jnp.sum(labels*jnn.log_softmax(logits, axis=0))


### Layer normalization
@jax.vmap
def layer_norm(X):  
    mean = jnp.mean(X, axis=0)
    var = jnp.var(X, axis=0)
    return (X - mean)/jnp.sqrt(var + 1e-7)

