from functools import partial
from tqdm import tqdm

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import optax

import graphax as gx

import tonic
from tonic import transforms

from torch.utils.data import DataLoader


batchsize = 16
num_channels = 700
N_hidden = 128
num_time_bins = 200
num_targets = 20 # 20 classes because numbers 0-9 are in English and German


# Load dataset with tonic
sensor_size = tonic.datasets.SHD.sensor_size
trans = transforms.Compose([transforms.ToFrame(sensor_size=sensor_size, 
                                                n_time_bins=num_time_bins)])

train_dataset = tonic.datasets.SHD(save_to="./data", train=True, transform=trans)
test_dataset = tonic.datasets.SHD(save_to="./data", train=False, transform=trans)

train_dataloader = DataLoader(train_dataset, 
                            batch_size=batchsize, 
                            shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

surrogate = lambda x: 1. / (1. + 10.*jnp.abs(x))
# Simple SNN model implementation according to the following paper:
# Zenke and Neftci, and Bellec et al.
# this is basically the f function of the recursion relation
# TODO: This shit needs a proper API!
def simple_SNN(x, z, v, W, V):
    beta = 0.95
    v_next = beta * v + (1. - beta) * (jnp.dot(W, x) + jnp.dot(V, z))
    # implementation of surrogate gradient without custom gradient features
    surr = surrogate(v_next)
    # TODO Jamie: fix heaviside function
    z_next = lax.stop_gradient(jnp.heaviside(v_next, 0.) - surr) + surr
    return z_next, v_next


def simple_SNN_stopgrad(x, z, v, W, V):
    beta = 0.95
    v_next = beta * v + (1. - beta) * (jnp.dot(W, x) + jnp.dot(V, z))
    # implementation of surrogate gradient without custom gradient features
    surr = surrogate(v_next)
    z_next = lax.stop_gradient(jnp.heaviside(v_next, 0.) - surr) + surr
    return z_next, v_next


# Use cross-entropy loss
def loss_fn(z, tgt, W_out):
    out = W_out @ z
    probs = jnn.softmax(out) # TODO Jamie: why does log_softmax not work in graphax?
    return -jnp.dot(tgt, jnp.log(probs)) # cross-entopy loss


# For loop that iterates over T time steps
# Accumulates the actual gradient G_t through recursion
# weight sharing -/-> trace sharing in convnets
# @partial(jax.jit, static_argnums=(5,))
def SNN_eprop_timeloop(xs, tgt, z0, v0, W, V, W_out, G_W0, G_V0):
    # G should be materalized as a PyTree of sparse tensors
    def loop_fn(carry, xs):
        z, v, G_W_val, G_V_val, W_grad_val, V_grad_val, W_out_grad_val, loss = carry
        # TODO Jamie: This is inefficient because we effectively evaluate simple_SNN twice!
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
        
        loss += loss_fn(next_z, tgt, W_out) # TODO Jamie: Again, we need to remove the inefficiency here!
        loss_grads = gx.jacve(loss_fn, order="rev", argnums=(0,2), sparse_representation=True)(next_z, tgt, W_out)
        loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
        W_grad = loss_grad * G_W
        V_grad = loss_grad * G_V

        W_grad_val += W_grad.val
        V_grad_val += V_grad.val
        W_out_grad_val += W_out_grad.val

        new_carry = (next_z, next_v, G_W.val, G_V.val, W_grad_val, V_grad_val, W_out_grad_val, loss)
        return new_carry, None
    
    # TODO: implement pytree handling for SparseTensor types
    final_carry, _ = lax.scan(loop_fn, (z0, v0, G_W0, G_V0, G_W0, G_V0, jnp.zeros((num_targets, size)), 0.), xs, length=num_time_bins)
    _, _, _, _, W_grad, V_grad, W_out_grad, loss = final_carry
    return loss, W_grad, V_grad, W_out_grad # final gradients


vmap_SNN_eprop_timeloop = jax.vmap(SNN_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None, None))


def SNN_bptt_timeloop(xs, tgt, z0, v0, W, V, W_out):
    def loop_fn(carry, xs):
        z, v, loss = carry
        next_z, next_v = simple_SNN_stopgrad(xs, z, v, W, V)
        # By neglecting the gradient wrt. S, we basically compute only the 
        # implicit recurrence, but not the explicit recurrence
        loss += loss_fn(next_z, tgt, W_out)
        new_carry = (next_z, next_v, loss)
        return new_carry, None
    
    # TODO: implement pytree handling for SparseTensor types
    carry, _ = lax.scan(loop_fn, (z0, v0, 0.), xs, length=num_time_bins)
    z, v, loss = carry
    return loss # final gradients


vmap_SNN_bptt_timeloop = jax.vmap(SNN_bptt_timeloop, in_axes=(0, 0, None, None, None, None, None))


@partial(jax.jacrev, argnums=(4, 5, 6), has_aux=True)
def loss_and_grad(xs, target, z0, v0, _W, _V, _W_out):
    losses = vmap_SNN_bptt_timeloop(xs, target, z0, v0, _W, _V, _W_out)
    return jnp.mean(losses), jnp.mean(losses) # has to return this twice so that it returns loss and grad!


size = N_hidden
z0 = jnp.zeros(size)
v0 = jnp.zeros(size)

key = jrand.PRNGKey(0)
wkey, vkey, woutkey = jrand.split(key, 3)
W = jrand.normal(wkey, (size, num_channels))
V = jrand.normal(vkey, (size, size))
W_out = jrand.normal(woutkey, (num_targets, size))

G_W0 = jnp.zeros((size, num_channels)) # gx.sparse_tensor_zeros_like(grads[1])
G_V0 = jnp.zeros((size, size)) # gx.sparse_tensor_zeros_like(grads[2])

optim = optax.adam(1e-3)
weights = (W, V, W_out)
opt_state = optim.init(weights)

@jax.jit
def eprop_train_step(xs, target, opt_state, weights, G_W0, G_V0):
    _W, _V, _W_out = weights
    loss, W_grad, V_grad, W_out_grad = vmap_SNN_eprop_timeloop(xs, target, z0, v0, _W, _V, _W_out, G_W0, G_V0)
    grads = (W_grad.mean(), V_grad.mean(), W_out_grad.mean()) # take the mean across the batch dim for all gradient updates
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights, opt_state

@jax.jit
def bptt_train_step(xs, target, opt_state, weights):
    _W, _V, _W_out = weights
    grads, loss = loss_and_grad(xs, target, z0, v0, _W, _V, _W_out)
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights, opt_state


# NOTE training loop
for data, targets in tqdm(train_dataloader):
    xs = jnp.array(data.numpy()).squeeze()
    targets = jnp.array(targets.numpy())
    targets = jnn.one_hot(targets, num_targets)
    # just comment out 'bptt' with 'eprop' to switch between the two training methods
    loss, weights, opt_state = eprop_train_step(xs, targets, opt_state, weights, G_W0, G_V0)
    # loss, weights, opt_state = bptt_train_step(xs, targets, opt_state, weights)
    print("loss", loss.mean())


# TODO Anil implement test loop and accuracy computation here and in the loop above




# xs = jnp.ones((num_time_bins, size))
# tgt = jnp.ones((size,))


# eprop_grads = SNN_eprop_timeloop(xs, z0, v0, W, V, T, tgt, G_W0, G_V0)
# bptt_grad_fn =jax.jacrev(SNN_BPTT_timeloop, argnums=(3, 4))
# jitted_bptt_grad_fn = jax.jit(bptt_grad_fn, static_argnums=5)
# bptt_grads = jitted_bptt_grad_fn(xs, z0, v0, W, V, T, tgt)


# rtrl_grad_fn = jax.jacfwd(SNN_BPTT_timeloop, argnums=(3, 4))
# jitted_rtrl_grad_fn = jax.jit(rtrl_grad_fn, static_argnums=5)
# rtrl_grads = jitted_rtrl_grad_fn(xs, z0, v0, W, V, T, tgt)

# import time
# start = time.time()
# eprop_grads = SNN_eprop_timeloop(xs, z0, v0, W, V, T, tgt, G_W0, G_V0)
# print("E-prop time elapsed", time.time() - start)
# start = time.time()
# bptt_grads = jitted_bptt_grad_fn(xs, z0, v0, W, V, T, tgt)
# print("BPTT Time elapsed", time.time() - start)
# start = time.time()
# rtrl_grads = jitted_rtrl_grad_fn(xs, z0, v0, W, V, T, tgt)
# print("RTRL Time elapsed", time.time() - start)


# print("e-prop gradients", eprop_grads)
# print("bptt gradients", bptt_grads)

