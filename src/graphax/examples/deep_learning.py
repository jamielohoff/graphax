import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def make_softmax_attention():    
    def attn(q, k, v):
        a = q.T @ k
        z = jnn.softmax(a, axis=1)
        return z @ v
    
    q = jnp.ones((4, 4))
    k = jnp.ones((4, 4))
    v = jnp.ones((4, 4))
    
    print(jax.make_jaxpr(attn)(q, k, v))
    
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
    
