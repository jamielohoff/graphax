from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import graphax as gx


# Simple SNN model implementation according to the following paper:
# Zenke and Neftci, and Bellec et al.
# this is basically the f function of the recursion relation
# TODO: This shit needs a proper API!
def simple_SNN(x, z, v, W, V):
    beta = 0.95
    v_next = beta * v + (1. - beta) * (jnp.dot(W, x) + jnp.dot(V, z))
    z_next = jnp.tanh(v_next) # spiking neurons don't for now
    # TODO: Use Emre's tricks for surrgate gradient as long as its not implemented!
    return z_next, v_next

def simple_SNN_stopgrad(x, z, v, W, V):
    beta = 0.95
    v_next = beta * v + (1. - beta) * (jnp.dot(W, x) + jnp.dot(V, z))
    z_next = jnp.tanh(v_next) # spiking neurons don't for now
    # TODO: Use Emre's tricks for surrgate gradient as long as its not implemented!
    return z_next, v_next


def loss_fn(z, tgt):
    d = z - tgt
    return 0.5 * jnp.dot(d, d)


# For loop that iterates over T time steps
# Accumulates the actual gradient G_t through recursion
# weight sharing -/-> trace sharing in convnets
@partial(jax.jit, static_argnums=(5,))
def SNN_eprop_timeloop(xs, z0, v0, W, V, T, tgt, G_W0, G_V0):
    # G should be materalized as a pytree of sparse tensors

    def loop_fn(carry, xs):
        z, v, G_W_val, G_V_val = carry

        # TODO: This is inefficient because we effectively evaluate simple_SNN twice!
        next_z, next_v = simple_SNN(xs, z, v, W, V)
        grads = gx.jacve(simple_SNN, order="rev", argnums=(2, 3, 4), sparse_representation=True)(xs, z, v, W, V)
        grads = grads[1]
        # By neglecting the gradient wrt. z, we basically compute only the 
        # implicit recurrence, but not the explicit recurrence
        # Look at OTTT, OSTL, FPTT etc.
        # print("H_E", grads[0]) # This guy is dense! but we could make it sparse...
        F_W, F_V = grads[1], grads[2]
        G_W = F_W.copy(G_W_val)
        G_V = F_V.copy(G_V_val)

        H_I = grads[0] # gradient of the implicit recurrence
        G_W = H_I * G_W + F_W
        G_V = H_I * G_V + F_V
        
        new_carry = (next_z, next_v, G_W.val, G_V.val)

        loss_grad = gx.jacve(loss_fn, order="rev", argnums=(0,), sparse_representation=True)(next_z, tgt)
        W_grad = loss_grad * G_W
        V_grad = loss_grad * G_V
        out = (W_grad.val, V_grad.val)
        return new_carry, out
    
    # TODO: implement pytree handling for SparseTensor types
    _, out = lax.scan(loop_fn, (z0, v0, G_W0, G_V0), xs, length=T)
    W_grad, V_grad = out
    return jnp.sum(W_grad, axis=0), jnp.sum(V_grad, axis=0) # final gradients


def SNN_BPTT_timeloop(xs, z0, v0, W, V, T, tgt):
    def loop_fn(carry, xs):
        z, v, loss = carry
        next_z, next_v = simple_SNN_stopgrad(xs, z, v, W, V)
        # By neglecting the gradient wrt. S, we basically compute only the 
        # implicit recurrence, but not the explicit recurrence
        loss += loss_fn(next_z, tgt)
        new_carry = (next_z, next_v, loss)
        return new_carry, None
    
    # TODO: implement pytree handling for SparseTensor types
    carry, _ = lax.scan(loop_fn, (z0, v0, 0.), xs, length=T)
    z, v, loss = carry
    return loss # final gradients

T = 100_000
size = 128
xs = jnp.ones((T, size))
tgt = jnp.ones((size,))
z0 = jnp.zeros(size)
v0 = jnp.zeros(size)

key = jrand.PRNGKey(0)
wkey, vkey = jrand.split(key)
W = jrand.normal(wkey, (size, size))
V = jrand.normal(vkey, (size, size))

G_W0 = jnp.zeros((size, size)) # gx.sparse_tensor_zeros_like(grads[1])
G_V0 = jnp.zeros((size, size)) # gx.sparse_tensor_zeros_like(grads[2])

eprop_grads = SNN_eprop_timeloop(xs, z0, v0, W, V, T, tgt, G_W0, G_V0)
bptt_grad_fn =jax.jacrev(SNN_BPTT_timeloop, argnums=(3, 4))
jitted_bptt_grad_fn = jax.jit(bptt_grad_fn, static_argnums=5)
bptt_grads = jitted_bptt_grad_fn(xs, z0, v0, W, V, T, tgt)
rtrl_grad_fn = jax.jacfwd(SNN_BPTT_timeloop, argnums=(3, 4))
jitted_rtrl_grad_fn = jax.jit(rtrl_grad_fn, static_argnums=5)
rtrl_grads = jitted_rtrl_grad_fn(xs, z0, v0, W, V, T, tgt)

import time
start = time.time()
eprop_grads = SNN_eprop_timeloop(xs, z0, v0, W, V, T, tgt, G_W0, G_V0)
print("E-prop time elapsed", time.time() - start)
start = time.time()
bptt_grads = jitted_bptt_grad_fn(xs, z0, v0, W, V, T, tgt)
print("BPTT Time elapsed", time.time() - start)
start = time.time()
rtrl_grads = jitted_rtrl_grad_fn(xs, z0, v0, W, V, T, tgt)
print("RTRL Time elapsed", time.time() - start)


# print("e-prop gradients", eprop_grads)
# print("bptt gradients", bptt_grads)

