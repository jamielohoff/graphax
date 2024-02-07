from functools import partial
import tqdm

import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import numpy as np

import optax

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import graphax as gx

from _transformer import (multihead_attention_block, glorot, gelu,
                        make_positional_encoding, softmax_ce_loss)   


epochs = 25
batchsize = 256
num_heads = 8
dk = 256

tiling = (4, 4)
input_shape = (3, 32, 32)
embedding_dim = input_shape[0]*input_shape[1]*input_shape[2]//(tiling[0]*tiling[1])
seq_len = tiling[0]*tiling[1] + 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

# Generate weights for attention blocks and MLP layers
def make_weights(key, num_attn_blocks: int = 2):
    weights = []
    for _ in range(num_attn_blocks):
        # Weigths for self-attention
        qkey, kkey, vkey, okey, key = jrand.split(key, 5)
        WQ = glorot(qkey, (dk*num_heads, embedding_dim))
        WK = glorot(kkey, (dk*num_heads, embedding_dim))
        WV = glorot(vkey, (dk*num_heads, embedding_dim))
        WO = glorot(okey, (embedding_dim, dk*num_heads))
        
        # Weights for MLP layers
        W1key, W2key, key = jrand.split(key, 3)
        W1 = glorot(W1key, (1024, embedding_dim))
        b1 = jnp.zeros((1024,))
        W2 = glorot(W2key, (embedding_dim, 1024))
        b2 = jnp.zeros((embedding_dim,))
        
        weights.extend([WQ, WK, WV, WO, W1, b1, W2, b2])
    return weights
        

key = jrand.PRNGKey(42)
weights = make_weights(key, 2)

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
                            None, None, None, None))
def transformer(X, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2, 
                WQ2, WK2, WV2, WO2, W3, b3, W4, b4, W5, b5, W6, b6):
    X = jnp.concatenate((CT, X), axis=1)
    X = positional_encoding(X)
    X = multihead_attention_block(X, WQ1, WK1, WV1, WO1, W1, b1, W2, b2)
    X = multihead_attention_block(X, WQ2, WK2, WV2, WO2, W3, b3, W4, b4)
    
    X = X[:, 0]
    return W6 @ gelu(W5 @ X + b5) + b6


def model(X, labels, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2, 
                WQ2, WK2, WV2, WO2, W3, b3, W4, b4, W5, b5, W6, b6):
    out = transformer(X, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2, 
                    WQ2, WK2, WV2, WO2, W3, b3, W4, b4, W5, b5, W6, b6)
    return softmax_ce_loss(out, labels)


def batched_model(X, labels, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2, 
                WQ2, WK2, WV2, WO2, W3, b3, W4, b4, W5, b5, W6, b6):
    return model(X, labels, CT, WQ1, WK1, WV1, WO1, W1, b1, W2, b2, 
                WQ2, WK2, WV2, WO2, W3, b3, W4, b4, W5, b5, W6, b6).sum()
    

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
        
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
    return loss, weights

optim = optax.adam(3e-4)
opt_state = optim.init(weights)

### Training loop
pbar = tqdm.tqdm(range(epochs))
for epoch in pbar:
    for (batch, labels) in tqdm.tqdm(trainloader):
        batch = batch.numpy()
        labels = labels.numpy()
        loss, weights = train(batch, labels, weights, opt_state)
        
        pbar.set_description(f"loss: {loss}")
        
    if epoch % 1 == 0:
        accs = []
        for (batch, labels) in tqdm.tqdm(testloader):
            batch = batch.numpy()
            labels = labels.numpy()
            acc = get_test_accuracy(batch, labels, weights)
            accs.append(acc)
        print(f"Test accuracy: {np.mean(accs)}")
    
