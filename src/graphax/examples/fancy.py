import jax
import jax.nn as jnn
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def make_f():
    def f(x, y, z, w):
        a = x * y
        b = z - w
        c = jnp.sin(x) + jnp.cos(y)
        d = jnp.exp(z) / w
        e = a + b
        f = jnp.sum(x) + c
        g = jnp.log(d)
        h = jnp.sqrt(jnp.abs(e))
        i = jnp.tan(h)
        j = jnp.maximum(x, 0)
        return e, f, g, i

    return make_graph(f, 1., 1., 1., 1.)


def make_g(size: int = 5):
    def g(x):
        return jnp.sum(jnp.sin(x) * jnp.cos(x**2) + jnp.log(x) - x**3 + jnp.exp(x), axis=0)
    
    x = jnp.ones(size)
    return make_graph(g, x)


def make_minimal_reverse():
    # Define the component functions
    def g1(x, y, z, w):
        return x ** 2 + y ** 2

    def g2(x, y, z, w):
        return jnp.sin(z) + jnp.log(w)
    
    # Define the overall function
    def minimal_reverse(x, y, z, w):
        return g1(x, y, z, w) + g2(x, y, z, w)

    print(jax.make_jaxpr(minimal_reverse)(1., 1., 1., 1.))
    return make_graph(minimal_reverse, 1., 1., 1., 1.)


def make_hessian():
    def f(x, y, z, w):
        a = jnp.cos(y) * jnp.sin(z)
        b = jnp.exp(x + w)
        return a + b

    grad_f = jax.grad(f)
    return make_graph(grad_f, 1., 1., 1., 1.)


def make_softmax_attention():
    def softmax(x):
        exp = jnp.exp(x)
        norm = jnp.sum(exp, axis=-1)
        return exp / norm
    
    def attn(q, k, v):
        a1 = q[0] * k
        a2 = q[1] * k
        b1 = jnp.dot(softmax(a1), v)
        b2 = jnp.dot(softmax(a2), v)
        return jnp.array([b1, b2])
    
    q = jnp.ones(2)
    k = jnp.ones(2)
    v = jnp.ones(2)
    
    print(jax.make_jaxpr(attn)(q, k, v))
    
    return make_graph(attn, q, k, v)

