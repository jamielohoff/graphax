import jax.nn as jnn
import jax.numpy as jnp


def sigmoid(x):
    return 1. / (1. + jnp.exp(-x))


def SiLU(x):
    return x*sigmoid(x)


def softmax_cross_entropy(logits, labels):
    return -jnp.sum(labels * jnp.log(jnn.softmax(logits, axis=-1)), axis=-1)


def variance(x, mu, axis=0):
    n = x.shape[axis]
    return jnp.sum((x - mu)**2, axis=axis, keepdims=True)/n


def layer_norm(x, gamma, beta):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    sigma = variance(x, mu, axis=-1)
    return (x - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta


def attn(q, k, v):
    a = q @ k.T
    z = jnn.softmax(a, axis=-1)
    return z @ v


def Perceptron(x, y, W1, b1, W2, b2, gamma, beta):
    y1 = jnp.tanh(x @ W1 + b1)
    # y2 = layer_norm(y1, gamma, beta)
    y3 = jnp.tanh(y1 @ W2 + b2)
    # d = y3 - y
    # 0.5 * jnp.mean(d**2) # 
    return softmax_cross_entropy(y3, y)


def encoder_block(x, WQ, WK, WV, W, b, gamma, beta):
    q = x @ WQ
    k = x @ WK
    v = x @ WV
    
    a = x + attn(q, k, v)
    c = layer_norm(a, gamma, beta)
    return SiLU(c @ W + b)


def decoder_block(x, kenc, venc, WQ1, WK1, WV1, WQ2, WK2, WV2, W, b, gamma0, gamma1, beta0, beta1):
    q1 = x @ WQ1
    k1 = x @ WK1
    v1 = x @ WV1
    
    a1 = x + attn(q1, k1, v1)
    c1 = layer_norm(a1, gamma0, beta0)
    
    q2 = WQ2 @ c1
    
    a2 = c1 + attn(q2, kenc, venc)
    c2 = layer_norm(a2, gamma1, beta1)
    return SiLU(W @ c2 + b)
    

def Encoder(x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, gamma0, beta0, gamma1, beta1):
    z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, gamma0, beta0)
    z2 = encoder_block(z1, WQ2, WK2, WV2, W2, b2, gamma1, beta1)
    return softmax_cross_entropy(z2, y)
    

def EncoderDecoder(x, y, WQ1, WQ2, WQ3, WK1, WK2, WK3, WV1, WV2, WV3,  W1, W2, b1, b2, gamma0, beta0, gamma1, beta1, gamma2, beta2):
    z1 = encoder_block(x, WQ1, WK1, WV1, W1, b1, gamma0, beta0)
    z2 = decoder_block(x, z1, z1, WQ2, WQ3, WK2, WK3, WV2, WV3, W2, b2, gamma1, gamma2, beta1, beta2)
    return .5*(z2 - y)**2

