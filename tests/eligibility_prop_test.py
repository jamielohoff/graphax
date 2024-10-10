import jax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import graphax as gx


# Simple SNN model implementation according to the following paper:
# Zenke and Neftci, and Bellec et al.
# this is basically the f function of the recursion relation
def simple_SNN(S_in, S, U, W, V):
    beta = 0.9
    U_next = beta * U + (1 - beta) * (jnp.dot(W, S_in) + jnp.dot(V, S))
    S_next = jnp.tanh(U)
    return S_next, U_next


# For loop that iterates over T time steps
# Accumulates the actual gradient G_t through recursion
def SNN_timeloop(S_in, S, U, W, V, T):
    # G should be materalized as a pytree of sparse tensors
    G = None
    for t in range(T):
        S, U = simple_SNN(S_in, S, U, W, V)
        grads = gx.jacve(simple_SNN, order="rev", argnums=(1, 2, 3, 4), dense_representation=False)(S_in, S, U, W, V)
        if t == 0:
            G = jtu.tree_map(lambda x: gx.sparse_tensor_zeros_like(x) if x is not None else None, grads)
        else:
            G = jtu.tree_map(lambda x, y: y * x, G, grads)
        print(G)
    return S, U


print(jax.make_jaxpr(simple_SNN)(jnp.array([1., 2.]), 
                                jnp.array([1., 2.]), 
                                jnp.array([1., 2.]), 
                                jnp.array([[1., 2.], 
                                           [3., 4.]]), 
                                jnp.array([[1., 2.], 
                                           [3., 4.]])))


print(SNN_timeloop(jnp.array([1., 2.]),
                    jnp.array([1., 2.]),
                    jnp.array([1., 2.]),
                    jnp.array([[1., 2.],
                                [3., 4.]]),
                    jnp.array([[1., 2.],
                                [3., 4.]]),
                    3))

