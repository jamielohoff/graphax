import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax import jacve, tree_allclose
from graphax.examples.randoms import f, g


key = jrand.PRNGKey(42)

# def f(x, y):
#     x = jnp.sin(x)
#     x = lax.slice(x, start_indices=[0, 0], limit_indices=[2, 3])
#     return x * y

# xkey, ykey = jrand.split(key, 2)
# x = jrand.normal(xkey, (3,3))
# y = jrand.normal(ykey, (3,))

# jaxpr = jax.make_jaxpr(f)(x, y)
# print(jaxpr)

# jaxpr = jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y)
# print(jaxpr)

# deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
# veres = deriv_fn(x, y)

# revres = jax.jacrev(f, argnums=(0, 1))(x, y)

# print(veres)
# print(revres)

# print(tree_allclose(veres, revres))

# def f(x, y):
#     z = x @ y
#     return jnp.sin(z)

# xkey, ykey = jrand.split(key, 2)
# x = jrand.normal(xkey, (2, 3))
# y = jrand.normal(ykey, (3,))
# jaxpr = jax.make_jaxpr(jacve(f, order="rev", argnums=(0, 1)))(x, y)
# print(jaxpr)

# deriv_fn = jax.jit(jacve(f, order="rev", argnums=(0, 1)))
# veres = deriv_fn(x, y)

# revres = jax.jacrev(f, argnums=(0, 1))(x, y)

# print(veres)
# print(revres)

# print(tree_allclose(veres, revres))

# xs = [jnp.ones(())*0.01]*15
# argnums = list(range(15))
# # print(jax.make_jaxpr(g)(*xs))
# jac_rev_g = jax.jit(jacve(g, order="rev", argnums=argnums))
# jax_rev_g = jax.jit(jax.jacrev(g, argnums=argnums))

# print(len(jax.make_jaxpr(jax.jacrev(g, argnums=argnums))(*xs).eqns))
# print(len(jax.make_jaxpr(jacve(g, order="rev", argnums=argnums))(*xs).eqns))

# revres = jax_rev_g(*xs)
# veres = jac_rev_g(*xs)

# # for rev, ve in zip(revres, veres):
# #     print(rev, ve)

# print(tree_allclose(veres, revres))


# def f(x, y):
#     z = x * y
#     w = z**3
#     return w + z, 5*w

# x = jnp.ones((50, 50))
# y = jnp.ones((50, 50))
# jaxpr = jax.make_jaxpr(f)(x, y)
# print(jaxpr)

# jacrev_f = jax.jit(jacve(f, order="rev", argnums=(0, 1), count_ops=True))
# jaxpr = jax.make_jaxpr(jacrev_f)(x, y)
# print(jaxpr)
# veres = jacrev_f(x, y)
# print(veres)

# jac_f = jax.jit(jax.jacrev(f, argnums=(0, 1)))
# revres = jac_f(x, y)
# print(revres)
# print(tree_allclose(veres, revres))


# def Helmholtz(x):
#     return x*jnp.log(x / (1. + -jnp.sum(x)))

# x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(4)/2000. # 

# print(jax.make_jaxpr(Helmholtz)(x))

# jac_cc = jax.jit(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))
# # print(jax.make_jaxpr(jacve(Helmholtz, order=[2, 5, 4, 3, 1]))(x))
# veres = jac_cc(x)


# jax_jac_fwd = jax.jit(jax.jacfwd(Helmholtz))
# jax_jac_rev = jax.jit(jax.jacfwd(Helmholtz))
# revres = jax_jac_rev(x)

# # TODO management of vector derivatives and so on
# print(veres)
# print(revres)
# print(tree_allclose(veres, revres))

# print(revres)
# print(veres)

# def transpose(x, y):
#     return x.T + y

# x = jnp.ones((2, 3))
# y = jnp.ones((3, 2))
# # print(jax.make_jaxpr(jacve(transpose, [1]))(x, y))
# jac_fwd = jax.jit(jacve(transpose, order="fwd", argnums=(0, 1)))
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

# x = jnp.ones(4)
# y = jrand.normal(key, (4,))

# w1key, b1key, key = jrand.split(key, 3)
# W1 = jrand.normal(w1key, (8, 4))
# b1 = jrand.normal(b1key, (8,))


# w2key, b2key, key = jrand.split(key, 3)
# W2 = jrand.normal(w2key, (4, 8))
# b2 = jrand.normal(b2key, (4,))
# # print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))

# # print(jax.make_jaxpr(jacve(NeuralNetwork, order=[9, 8, 7, 6, 5, 4, 3, 2, 1]))(x, W1, b1, W2, b2, y))
# jac_rev = jax.jit(jacve(NeuralNetwork, order="rev", argnums=(0, 1, 2, 3, 4, 5)))
# veres = jac_rev(x, W1, b1, W2, b2, y)

# # print(jax.make_jaxpr(jax.jacrev(NeuralNetwork, argnums=(0, 1, 2, 3, 4, 5)))(x, W1, b1, W2, b2, y))
# jax_jac_rev = jax.jit(jax.jacrev(NeuralNetwork, argnums=(0, 1, 2, 3, 4, 5)))
# revres = jax_jac_rev(x, W1, b1, W2, b2, y)

# print(revres[1], veres[1])

# print(tree_allclose(veres, revres))

# from graphax.examples.randoms import f

# a = jnp.ones(4)
# b = jnp.ones((2, 3))
# c = jnp.ones((4, 4))
# d = jnp.ones((3, 3))
# e = jnp.ones((4, 1))
# xs = [a, b, c, d, e]


# jaxpr = jax.make_jaxpr(f)(*xs)
# print(jaxpr)

# deriv_fn = jax.jit(jacve(f, order="fwd", argnums=(0, 1)))
# veres = deriv_fn(*xs)

# revres = jax.jacrev(f, argnums=(0, 1))(*xs)

# print(veres)
# print(revres)

# print(tree_allclose(veres, revres))


def softmax_attention(X, WQ, WK, WV):
    q = WQ @ X
    k = WK @ X
    v = WV @ X
    a = q @ k.T
    return jnn.softmax(a, axis=1) @ v
 
x = jrand.normal(key, (10, 16))
WQ = jrand.normal(key,(10, 10)) 
WK = jrand.normal(key,(10, 10))
WV = jrand.normal(key,(10, 10))

print(jax.make_jaxpr(softmax_attention)(x, WQ, WK, WV))
print(jax.make_jaxpr(jacve(softmax_attention, order="fwd", argnums=(1, 2, 3)))(x, WQ, WK, WV))

