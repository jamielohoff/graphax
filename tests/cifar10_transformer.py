from functools import partial
import time
import tqdm

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import optax

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import graphax as gx

from _transformer import (multihead_attention_block, glorot, gelu,
                        make_positional_encoding, softmax_ce_loss,
                        make_weights)   


epochs = 25
batchsize = 4
num_heads = 8 # this seems to impact performance a lot
dk = 64

tiling = (4, 4)
input_shape = (3, 32, 32)
embedding_dim = input_shape[0]*input_shape[1]*input_shape[2]//(tiling[0]*tiling[1])
seq_len = tiling[0]*tiling[1] + 1

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), 
                                                    (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root="./data", train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)

testset = datasets.CIFAR10(root="./data", train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)

key = jrand.PRNGKey(42)
weights = make_weights(key, 3, dk, num_heads, embedding_dim)

# Weights for classification head
W5key, W6key, key = jrand.split(key, 3)
W5 = glorot(W5key, (256, embedding_dim))
b5 = jnp.zeros(256, dtype=jnp.float32)
W6 = glorot(W6key, (10, 256))
b6 = jnp.zeros(10, dtype=jnp.float32)

# Weights for class token
CT = jrand.normal(key, (embedding_dim, 1))

weights = [CT] + weights + [W5, b5, W6, b6]
weights = tuple(weights)

### Positional encoding
positional_encoding = make_positional_encoding(seq_len, embedding_dim)

### Transformer model
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, 
                            None, None, None, None, None, None, None, None, None,
                            None, None, None, None, None, None))
def transformer(X, CT, WQKV1, WO1, W1, b1, W2, b2, 
                        WQKV2, WO2, W3, b3, W4, b4, 
                        WQKV3, WO3, W5, b5, W6, b6,
                        W7, b7, W8, b8):
    X = jnp.concatenate((CT, X), axis=1)
    X = positional_encoding(X)
    X = multihead_attention_block(X, WQKV1, WO1, W1, b1, W2, b2)
    # X = multihead_attention_block(X, WQKV2, WO2, W3, b3, W4, b4)
    # X = multihead_attention_block(X, WQKV3, WO3, W5, b5, W6, b6)
    
    X = X[:, 0]
    return W8 @ gelu(W7 @ X + b7) + b8


def model(X, labels, CT, WQKV1, WO1, W1, b1, W2, b2, 
                        WQKV2, WO2, W3, b3, W4, b4, 
                        WQKV3, WO3, W5, b5, W6, b6,
                        W7, b7, W8, b8):
    out = transformer(X, CT, WQKV1, WO1, W1, b1, W2, b2, 
                            WQKV2, WO2, W3, b3, W4, b4, 
                            WQKV3, WO3, W5, b5, W6, b6,
                            W7, b7, W8, b8)
    return softmax_ce_loss(out, labels)


def batched_model(X, labels, CT, WQKV1, WO1, W1, b1, W2, b2, 
                                WQKV2, WO2, W3, b3, W4, b4, 
                                WQKV3, WO3, W5, b5, W6, b6,
                                W7, b7, W8, b8):
    return model(X, labels, CT, WQKV1, WO1, W1, b1, W2, b2, 
                                WQKV2, WO2, W3, b3, W4, b4, 
                                WQKV3, WO3, W5, b5, W6, b6,
                                W7, b7, W8, b8).sum()
    

### Function to subdivide image into tiles for vision transformer
@jax.vmap
def subdivide(img, tiles = (4, 4)):
    img = jnp.moveaxis(img, 0, -1)
    grid = [jnp.hsplit(half, tiles[0]) for half in jnp.vsplit(img, tiles[1])]
    return jnp.array([h.flatten() for g in grid for h in g]).transpose(1, 0)


### Function to evaluate the model on the test dataset
@jax.jit
def get_test_accuracy(batch, label, weights):
    xs = subdivide(batch)
    pred = jnp.argmax(transformer(xs, *weights), axis=1)
    return jnp.sum(pred == label)/pred.shape[0]


### Train function
@jax.jit
def train(batch, labels, weights, opt_state):
    one_hot_labels = jnn.one_hot(labels, 10)
    xs = subdivide(batch)
    argnums = range(2, len(weights) + 2)
    loss = batched_model(xs, one_hot_labels, *weights)

    grads = gx.jacve(batched_model, order="rev", argnums=argnums)(xs, one_hot_labels, *weights)
    _grads = jax.jacrev(batched_model, argnums=argnums)(xs, one_hot_labels, *weights)
            
    close = gx.tree_allclose(grads, _grads)
    # jax.debug.print("Close: {x}", x=close)
    
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights

optim = optax.adam(1e-4)
opt_state = optim.init(weights)

batch, labels = next(iter(trainloader))
batch = batch.numpy()
labels = labels.numpy()

one_hot_labels = jnn.one_hot(labels, 10)
xs = subdivide(batch)
argnums = range(2, len(weights) + 2)

# print(jax.make_jaxpr(batched_model)(xs, one_hot_labels, *weights))

#print(jax.make_jaxpr(gx.jacve(batched_model, order="rev", argnums=argnums))(xs, one_hot_labels, *weights))
# bxh
# print(jax.make_jaxpr(jax.jacrev(batched_model, argnums=argnums))(xs, one_hot_labels, *weights))
# bfv

### Training loop
st = time.time()
pbar = tqdm.tqdm(range(epochs))
for epoch in pbar:
    for (batch, labels) in tqdm.tqdm(trainloader):
        batch = batch.numpy()
        labels = labels.numpy()
        loss, weights = train(batch, labels, weights, opt_state)
        pbar.set_description(f"loss: {loss:.4f}")
        
    if epoch % 1 == 0:
        accs = []
        for (batch, labels) in testloader:
            batch = batch.numpy()
            labels = labels.numpy()
            acc = get_test_accuracy(batch, labels, weights)
            accs.append(acc)
        print(f"Test accuracy: {jnp.mean(accs)*100:.2f}%")
print(f"Training took {time.time() - st:.2f} seconds")
    
