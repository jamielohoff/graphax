import time
import timeit

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.interpreter import jacve, tree_allclose

key = jrand.PRNGKey(42)

def f(x, y):
    z = x @ y
    return jnp.sin(z)

xkey, ykey = jrand.split(key, 2)
x = jrand.normal(xkey, (2, 3))
y = jrand.normal(ykey, (3,))
jaxpr = jax.make_jaxpr(jacve(f, [1]))(x, y)
print(jaxpr)

deriv_fn = jax.jit(jacve(f, [1], argnums=(0, 1)))
veres = deriv_fn(x, y)

revres = jax.jacrev(f, argnums=(0, 1))(x, y)

print(veres)
print(revres)

print(tree_allclose(veres, revres))


def f(x, y):
    z = x * y
    w = z**3
    return w + z, 5*w

x = 1. # jnp.ones((50, 50))
y = 2. # *jnp.ones((50, 50))
jaxpr = jax.make_jaxpr(jacve(f, [2, 1]))(x, y)
print(jaxpr)

jacs = jax.jit(jacve(f, [2, 1], argnums=(0, 1)))
veres = jacs(x, y)
print(veres)

jacfwd_f = jax.jit(jacve(f, [1, 2]))
jacrev_f = jax.jit(jacve(f, [2, 1]))
# print(timeit.timeit(lambda: jacfwd_f(x, y), number=1000))
# print(timeit.timeit(lambda: jacrev_f(x, y), number=1000))


jac_f = jax.jit(jax.jacfwd(f, argnums=(0, 1)))
# print(timeit.timeit(lambda: jac_f(x, y), number=1000))
jac_f = jax.jit(jax.jacrev(f, argnums=(0, 1)))
# # print(jax.make_jaxpr(jax.jacrev(f, argnums=(0, 1)))(x, y))
revres = jac_f(x, y)
# print(timeit.timeit(lambda: jac_f(x, y), number=1000))

# jac_f = jax.jacrev(f, argnums=(0, 1))
# jac_f(x, y)
# print(timeit.timeit(lambda: jac_f(x, y), number=1000))

print(tree_allclose(veres, revres))


def Helmholtz(x):
    return x*jnp.log(x / (1. + -jnp.sum(x)))

x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(4)/2000. # 
jac_fwd = jax.jit(jacve(Helmholtz, order="fwd"))
jac_rev = jax.jit(jacve(Helmholtz, order="rev"))
jac_cc = jax.jit(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))
# print(jax.make_jaxpr(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))(x))
veres = jac_cc(x)

# print(timeit.timeit(lambda: jac_cc(x), number=10000))
# print(timeit.timeit(lambda: jac_fwd(x), number=10000))
# print(timeit.timeit(lambda: jac_rev(x), number=10000))

jax_jac_fwd = jax.jit(jax.jacfwd(Helmholtz))
jax_jac_rev = jax.jit(jax.jacfwd(Helmholtz))
revres = jax_jac_rev(x)

# print(timeit.timeit(lambda: jax_jac_fwd(x), number=10000))
# print(timeit.timeit(lambda: jax_jac_rev(x), number=10000))

# TODO management of vector derivatives and so on
print(tree_allclose(veres, revres))

print(veres)
print(revres)

# def transpose(x, y):
#     return x.T + y

# x = jnp.ones((2, 3))
# y = jnp.ones((3, 2))
# # print(jax.make_jaxpr(jacve(transpose, [1]))(x, y))
# jac_fwd = jacve(transpose, order="fwd")
# veres = jac_fwd(x, y)[0]
# print(veres)

# revres = jax.jacrev(transpose)(x, y)
# print(revres)

# print(tree_allclose(veres, revres))

# def NeuralNetwork(x, W1, b1, W2, b2, y):
#     y1 = W1 @ x
#     z1 = y1 + b1
#     a1 = jnp.tanh(z1)
    
#     y2 = W2 @ a1
#     z2 = y2 + b2
#     a2 = jnp.tanh(z2)
#     d = a2 - y
#     return .5*jnp.sum(d**2)


# key = jrand.PRNGKey(42)

# x = jnp.ones(40)
# y = jrand.normal(key, (40,))

# w1key, b1key, key = jrand.split(key, 3)
# W1 = jrand.normal(w1key, (80, 40))
# b1 = jrand.normal(b1key, (80,))


# w2key, b2key, key = jrand.split(key, 3)
# W2 = jrand.normal(w2key, (40, 80))
# b2 = jrand.normal(b2key, (40,))
# # print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))

# # print(jax.make_jaxpr(jacve(NeuralNetwork, order=[9, 8, 7, 6, 5, 4, 3, 2, 1]))(x, W1, b1, W2, b2, y))
# jac_rev = jax.jit(jacve(NeuralNetwork, order="rev"))
# veres = jac_rev(x, W1, b1, W2, b2, y)[1]

# # print(jax.make_jaxpr(jax.jacrev(NeuralNetwork, argnums=(0, 1, 2, 3, 4, 5)))(x, W1, b1, W2, b2, y))
# jax_jac_rev = jax.jit(jax.jacrev(NeuralNetwork, argnums=(0, 1, 2, 3, 4, 5)))
# revres = jax_jac_rev(x, W1, b1, W2, b2, y)[1]

# print(jnp.allclose(veres, revres))

# st = time.time()
# for _ in range(1000):
#     key, subkey = jrand.split(key, 2)
#     x = jrand.normal(subkey, (40,))
#     grad = jac_rev(x, W1, b1, W2, b2, y)[1]
# print(time.time() - st)

# st = time.time()
# for _ in range(1000):
#     key, subkey = jrand.split(key, 2)
#     x = jrand.normal(subkey, (40,))
#     grad = jax_jac_rev(x, W1, b1, W2, b2, y)[1]
# print(time.time() - st)

# import jax.lax as lax
# import jax.nn as jnn


# def softmax_attention(X, WQ, WK, WV):
#     q = WQ @ X
#     k = WK @ X
#     v = WV @ X
#     a = q @ k.T
#     return jnn.softmax(a, axis=1) @ v
 
# x = jrand.normal(key, (10, 16))
# WQ = jrand.normal(key,(10, 10)) 
# WK = jrand.normal(key,(10, 10))
# WV = jrand.normal(key,(10, 10))

# print(jax.make_jaxpr(jacve(softmax_attention, order="fwd"))(x, WQ, WK, WV))

# def f(x, y):
#     z = x * y
#     return 3.0*z

# x = jnp.ones((2, 3))
# y = jnp.ones((3,))
# jaxpr = jax.make_jaxpr(jacve(f, "fwd"))(x, y)
# print(jaxpr)

# jacs = jacve(f, "fwd")(x, y)

# jax_jacs = jax.jacrev(f, argnums=(0, 1, 2, 3))(x, y)

# print(jnp.allclose(veres, revres))

# print("ve",jacs[0])
# print("rev", jax_jacs[0])

