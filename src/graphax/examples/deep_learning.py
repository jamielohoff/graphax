import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph

def attn(q, k, v):
    a = q.T @ k
    z = jnn.softmax(a, axis=1)
    return z @ v

def make_softmax_attention():    
    q = jnp.ones((4, 4))
    k = jnp.ones((4, 4))
    v = jnp.ones((4, 4))
    
    return make_graph(attn, q, k, v)


def make_Perceptron():
    def Perceptron(x, W1, b1, W2, b2, y):
        y1 = W1 @ x
        z1 = y1 + b1
        a1 = jnp.tanh(z1)
        
        y2 = W2 @ a1
        z2 = y2 + b2
        a2 = jnp.tanh(z2)
        d = a2 - y
        e = d**2
        return .5*jnp.sum(e)

    x = jnp.ones(4)
    y = jnp.ones(4)

    W1 = jnp.ones((3, 4))
    b1 = jnp.ones(3)

    W2 = jnp.ones((4, 3))
    b2 = jnp.ones(4)
    
    return make_graph(Perceptron, x, W1, b1, W2, b2, y)


def layer_norm(x, gamma, beta):
    mu = jnp.mean(x, axis=1)
    sigma = jnp.var(x, axis=1)
    
    return (x - mu)/jnp.sqrt(sigma + 1e-6) * gamma + beta


# TODO this might not be correct yet!
def make_transformer_encoder():
    def transformer(x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, gamma1, gamma2, beta1, beta2):
        q1 = WQ1 @ x
        k1 = WK1 @ x
        v1 = WV1 @ x
        
        a1 = x + attn(q1, k1, v1)
        c1 = layer_norm(a1, gamma1, beta1)
        z1 = jnp.tanh(W1 @ c1 + b1)
        
        q2 = WQ2 @ z1
        k2 = WK2 @ z1
        v2 = WV2 @ z1
        
        a2 = z1 + attn(q2, k2, v2)
        c2 = layer_norm(a2, gamma2, beta2)
        z2 = jnp.tanh(W2 @ c2 + b2)
        
        return .5*(z2 - y)**2
    
    x = jnp.ones((4, 4))
    y = jnp.ones(4)
    
    WQ1 = jnp.ones((4, 4))
    WK1 = jnp.ones((4, 4))
    WV1 = jnp.ones((4, 4))
    
    WQ2 = jnp.ones((4, 4))
    WK2 = jnp.ones((4, 4))
    WV2 = jnp.ones((4, 4))

    W1 = jnp.ones((4, 4))
    b1 = jnp.ones(4)

    W2 = jnp.ones((4, 4))
    b2 = jnp.ones(4)
        
    return make_graph(transformer, x, y, WQ1, WQ2, WK1, WK2, WV1, WV2, W1, W2, b1, b2, 1., 1., 0., 0.) 
    
