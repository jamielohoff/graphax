import graphax as gx
import jax
import jax.numpy as jnp
import jax.random as jrand
import equinox as eqx
import tonic
import torch
import optax

from functools import partial
from tqdm import tqdm
from neuron_models import SNN_LIF

# Handling the dataset:
BATCH_SIZE = 128
NUM_TIMESTEPS = 250
EPOCHS = 100
NUM_HIDDEN = 256
NUM_LABELS = 20
NUM_CHANNELS = 700
SENSOR_SIZE = tonic.datasets.SHD.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=SENSOR_SIZE, 
                                            n_time_bins=NUM_TIMESTEPS)

transform = tonic.transforms.Compose([frame_transform,])
train_set = tonic.datasets.SHD(save_to='./data', train=True, transform=transform)
test_set = tonic.datasets.SHD(save_to='./data', train=False, transform=transform)

torch.manual_seed(42)
train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

# Use cross-entropy loss
def loss_fn(z, tgt, W_out):
    out = W_out @ z
    probs = jax.nn.softmax(out) 
    log_probs = jnp.log(probs + 1e-8)
    # return optax.softmax_cross_entropy(out, tgt)
    return -jnp.dot(tgt, log_probs) # cross-entopy loss
    
'''
G: Accumulated gradient of U (hidden state) w.r.t. parameters W and V.
F: Immediate gradient of U (hidden state) w.r.t. parameters W and V.
H: Gradient of current U w.r.t. previous timestep U.
in_seq: (batch_dim, num_timesteps, sensor_size)
'''

def SNN_eprop_timeloop(in_seq, target, z0, u0, W, V, W_out, G_W0, G_V0):
    # NOTE: we have a vanishing gradient problem here!
    def loop_fn(carry, in_seq):
        z, u, G_W_val, G_V_val, W_grad_val, V_grad_val, W_out_grad_val, loss = carry
        outputs, grads = gx.jacve(SNN_LIF, order = 'rev', argnums=(2, 3, 4), has_aux=True, sparse_representation=True)(in_seq, z, u, W, V)
        next_z, next_u = outputs
        # grads contain the gradients of z_next and u_next w.r.t. u, W and V. But we ignore grads of z (explicit recurrence) 
        u_grads = grads[1] # only gradient of u_next w.r.t. u, W and V.
        # W is the 
        F_W, F_V = u_grads[1], u_grads[2] # gradients of u_next w.r.t. W and V respectively.
        G_W = F_W.copy(G_W_val) # G_W_val is the gradient of prev. timestep u w.r.t. W.
        G_V = F_V.copy(G_V_val) # G_V_val is the gradient of prev. timestep u w.r.t. V.

        H_I = u_grads[0] # grad. of u_next w.r.t. previous timestep u.
        G_W = H_I * G_W + F_W
        G_V = H_I * G_V + F_V

        _loss, loss_grads = gx.jacve(loss_fn, order = 'rev', argnums=(0, 2), has_aux=True, sparse_representation=True)(next_z, target, W_out)
        loss += _loss
        loss_grad, W_out_grad = loss_grads[0], loss_grads[1]
        W_grad = loss_grad * G_W
        V_grad = loss_grad * G_V

        W_grad_val += W_grad.val
        V_grad_val += V_grad.val
        W_out_grad_val += W_out_grad.val

        new_carry = (next_z, next_u, G_W.val, G_V.val, W_grad_val, V_grad_val, W_out_grad_val, loss)
        return new_carry, None
    final_carry, _ = jax.lax.scan(loop_fn, (z0, u0, G_W0, G_V0, G_W0, G_V0, jnp.zeros((NUM_LABELS, NUM_HIDDEN)), .0), in_seq, length=NUM_TIMESTEPS)
    _, _, _, _, W_grad, V_grad, W_out_grad, loss = final_carry
    return loss, W_grad, V_grad, W_out_grad

batch_vmap_SNN_eprop_timeloop = jax.vmap(SNN_eprop_timeloop, in_axes=(0, 0, None, None, None, None, None, None, None))

z0 = jnp.zeros(NUM_HIDDEN)
u0 = jnp.zeros(NUM_HIDDEN)

key = jrand.PRNGKey(0)
wkey, vkey, G_W_key, G_V_key, woutkey = jrand.split(key, 5)

def xavier_normal(key, shape):
    # Calculate the standard deviation for Xavier normal initialization
    fan_in, fan_out = shape
    stddev = jnp.sqrt(2.0 / (fan_in + fan_out))
    
    # Generate random numbers from a normal distribution
    return stddev * jax.random.normal(key, shape)

init_ = jax.nn.initializers.orthogonal() #  xavier_normal # jax.nn.initializers.he_normal()

W = init_(wkey, (NUM_HIDDEN, NUM_CHANNELS))
V = init_(vkey, (NUM_HIDDEN, NUM_HIDDEN))
W_out = init_(woutkey, (NUM_LABELS, NUM_HIDDEN))

G_W0 = init_(G_W_key, (NUM_HIDDEN, NUM_CHANNELS))
G_V0 = init_(G_V_key, (NUM_HIDDEN, NUM_HIDDEN))

optim = optax.adamax(1e-3)
weights = (W, V, W_out)
opt_state = optim.init(weights)

def SNN_bptt_timeloop(in_seq, tgt, z0, u0, W, V, W_out):
    def loop_fn(carry, in_seq):
        z, u, loss = carry
        next_z, next_u = SNN_LIF(in_seq, z, u, W, V)
        # By neglecting the gradient wrt. S, we basically compute only the 
        # implicit recurrence, but not the explicit recurrence
        loss += loss_fn(next_z, tgt, W_out)
        new_carry = (next_z, next_u, loss)
        return new_carry, None

    carry, _ = jax.lax.scan(loop_fn, (z0, u0, 0.), in_seq, length=NUM_TIMESTEPS)
    z, v, loss = carry
    return loss 


batch_vmap_SNN_bptt_timeloop = jax.vmap(SNN_bptt_timeloop, in_axes=(0, 0, None, None, None, None, None))


@partial(jax.jacrev, argnums=(4, 5, 6), has_aux=True)
def bptt_loss_and_grad(in_seq, target, z0, u0, _W, _V, _W_out):
    # losses = batch_vmap_SNN_eprop_timeloop(in_seq, target, z0, u0, _W, _V, _W_out)
    losses = batch_vmap_SNN_bptt_timeloop(in_seq, target, z0,u0, _W, _V, _W_out)
    loss = jnp.mean(losses)
    return loss, loss

# Train for one batch:
@jax.jit
def eprop_train_step(in_batch, target, opt_state, weights, G_W0, G_V0):
    _W, _V, _W_out = weights
    loss, W_grad, V_grad, W_out_grad = batch_vmap_SNN_eprop_timeloop(in_batch, target, z0, u0, _W, _V, _W_out, G_W0, G_V0)
    grads = (W_grad.mean(0), V_grad.mean(0), W_out_grad.mean(0)) # take the mean across the batch dim for all gradient updates
    updates, opt_state = optim.update(grads, opt_state)
    weights = jax.tree_util.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights, opt_state


@jax.jit
def bptt_train_step(in_seq, target, opt_state, weights):
    _W, _V, _W_out = weights
    grads, loss = bptt_loss_and_grad(in_seq, target, z0, u0, _W, _V, _W_out)
    updates, opt_state = optim.update(grads, opt_state)
    weights = jax.tree_util.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights, opt_state


def predict(in_seq, weights):
    W, V, W_out = weights
    def loop_fn(carry, in_seq):
        z, u = carry
        z_next, u_next = SNN_LIF(in_seq, z, u, W, V)
        carry = (z_next, u_next)
        return carry, None

    carry_final, _ = jax.lax.scan(loop_fn, (z0, u0), in_seq) # loop over the timesteps
    z_final = carry_final[0]
    # print('carry_final: ', carry_final)
    out = W_out @ z_final
    probs = jax.nn.softmax(out)
    return jnp.argmax(probs, axis=0)

# Test for one batch:

def test_step(in_batch, target_batch, weights):
    preds_batch = jax.vmap(predict, in_axes=(0, None))(in_batch, weights) # vmap over batch dimension
    # print('preds_batch: ', preds_batch)
    # print('target_batch: ', target_batch)
    return (preds_batch == target_batch).mean()

# Test loop
def test_model(weights):
    accuracy_batch = 0.
    num_iters = len(test_loader)
    for data, target_batch in tqdm(test_loader):
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        accuracy_batch += test_step(in_batch, target_batch, weights)
    accuracy = accuracy_batch / num_iters
    print('Accuracy: ', accuracy)

# Training loop
for ep in range(EPOCHS):
    for data, target_batch in tqdm(train_loader):
        in_batch = jnp.array(data.numpy()).squeeze()
        target_batch = jnp.array(target_batch.numpy())
        target_batch = jax.nn.one_hot(target_batch, NUM_LABELS)

        # just comment out 'bptt' with 'eprop' to switch between the two training methods
        loss, weights, opt_state = eprop_train_step(in_batch, target_batch, opt_state, weights, G_W0, G_V0)
        # loss, weights, opt_state = bptt_train_step(in_batch, target_batch, opt_state, weights)
        print("Epoch: ", ep + 1, ", loss: ", loss.mean() / NUM_TIMESTEPS)
    test_model(weights)
    
