import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_random_code


# def simple(x, y):
#         z = x * y
#         w = jnp.sin(z)
#         return w + z, jnp.log(w)

# print(jax.make_jaxpr(simple)(1., 1.))
# edges = make_graph(simple, 1., 1.)
# print(edges)


# def Helmholtz(x):
#     e = jnp.sum(x)
#     f = 1. + -e
#     w = x / f
#     z = jnp.log(w)
#     return x*z

# x = jnp.ones(4)
# print(jax.make_jaxpr(Helmholtz)(x))
# edges = make_graph(Helmholtz, x)
# print(edges)


# def NeuralNetwork(x, W1, b1, W2, b2, y):
#     y1 = W1 @ x
#     z1 = y1 + b1
#     a1 = jnp.tanh(z1)
    
#     y2 = W2 @ a1
#     z2 = y2 + b2
#     a2 = jnp.tanh(z2)
#     d = a2 - y
#     e = d**2
#     return .5*jnp.sum(e)

# x = jnp.ones(4)
# y = jnp.ones(4)

# W1 = jnp.ones((3, 4))
# b1 = jnp.ones(3)

# W2 = jnp.ones((4, 3))
# b2 = jnp.ones(4)
# print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))
# edges = make_graph(NeuralNetwork, x, W1, b1, W2, b2, y)
# print(edges)

import sys
jnp.set_printoptions(threshold=sys.maxsize)

key = jrand.PRNGKey(42)
code, jaxpr = make_random_code(key, [5, 15, 5])
print(code)
print(jaxpr)
edges = make_graph(jaxpr)
print(edges)

