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

from _transformer import (multihead_softmax_attention, MLP, 
                        layer_norm, make_positional_encoding,
                        softmax_ce_loss, glorot, gelu)   


epochs = 1000
batchsize = 256
num_heads = 6
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


# Weigths for first self-attention layer
key = jrand.PRNGKey(42)
qkey, kkey, vkey, okey, key = jrand.split(key, 5)
WQ1 = glorot(qkey, (dk*num_heads, embedding_dim))
WK1 = glorot(kkey, (dk*num_heads, embedding_dim))
WV1 = glorot(vkey, (dk*num_heads, embedding_dim))
WO1 = glorot(okey, (embedding_dim, dk*num_heads))

# Weights for second self-attention layer
qkey, kkey, vkey, okey, key = jrand.split(key, 5)
WQ2 = glorot(qkey, (dk*num_heads, embedding_dim))
WK2 = glorot(kkey, (dk*num_heads, embedding_dim))
WV2 = glorot(vkey, (dk*num_heads, embedding_dim))
WO2 = glorot(okey, (embedding_dim, dk*num_heads))

# Weights for MLP layers
W1key, W2key, key = jrand.split(key, 3)
W1 = glorot(W1key, (1024, embedding_dim))
b1 = jnp.zeros((1024, 1))
W2 = glorot(W2key, (embedding_dim, 1024))
b2 = jnp.zeros((embedding_dim, 1))

W3key, W4key, key = jrand.split(key, 3)
W3 = glorot(W3key, (1024, embedding_dim))
b3 = jnp.zeros((1024, 1), dtype=jnp.float32)
W4 = glorot(W4key, (embedding_dim, 1024))
b4 = jnp.zeros((embedding_dim, 1), dtype=jnp.float32)

W5key, W6key, key = jrand.split(key, 3)
W5 = glorot(W5key, (256, embedding_dim))
b5 = jnp.zeros(256, dtype=jnp.float32)
W6 = glorot(W6key, (10, 256))
b6 = jnp.zeros(10, dtype=jnp.float32)

# Weights for class token
CT = jrand.normal(key, (embedding_dim, 1))


### Positional encoding
positional_encoding = make_positional_encoding(seq_len, embedding_dim)


### Transformer model
@partial(jax.vmap, in_axes=(0, None, None, None, None, None, None, None, None, 
                            None, None, None, None, None, None, None, None, None,
                            None, None, None, None))
def transformer(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
                W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT):
    X = jnp.concatenate((CT, X), axis=1)
    X = positional_encoding(X)
    
    X += multihead_softmax_attention(X, WQ1, WK1, WV1, WO1)
    X = layer_norm(X)
    X += MLP(X, W1, b1, W2, b2)
    X = layer_norm(X)
    
    X += multihead_softmax_attention(X, WQ2, WK2, WV2, WO2)
    X = layer_norm(X)
    X += MLP(X, W3, b3, W4, b4)
    X = layer_norm(X)
    
    X = X[:, 0]
    return W6 @ gelu(W5 @ X + b5) + b6


def model(X, label, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
            W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT):
    out = transformer(X, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
                        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT)
    return softmax_ce_loss(out, label)


def batched_model(X, labels, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
            W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT):
    return model(X, labels, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
                        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT).sum()
    

### Function to subdivide image into tiles for vision transformer
@jax.vmap
def subdivide(img, tiles = (4, 4)):
    img = jnp.moveaxis(img, 0, -1)
    grid = [jnp.hsplit(half, tiles[0]) for half in jnp.vsplit(img, tiles[1])]
    return jnp.array([h.flatten() for g in grid for h in g]).transpose(1, 0)


### Function to evaluate the model on the test dataset
@jax.jit
def get_accuracy(batch, label, weights):
    xs = subdivide(batch)
    pred = jnp.argmax(transformer(xs, *weights), axis=1)
    return jnp.sum(pred == label)/pred.shape[0]


### Training loop
@jax.jit
def train(batch, labels, weights, opt_state):
    one_hot_labels = jnn.one_hot(labels, 10)
    xs = subdivide(batch)
    argnums = range(2, 23)
    loss = batched_model(xs, one_hot_labels, *weights)
    grads = jax.jacrev(batched_model, argnums=argnums)(xs, one_hot_labels, *weights)
    
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)

    return loss, weights

### Preparing optimizer and weights
weights = (WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
            W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, CT)

optim = optax.adam(3e-4)
opt_state = optim.init(weights)

# Training loop
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
            acc = get_accuracy(batch, labels, weights)
            accs.append(acc)
        print(f"Test accuracy: {np.mean(accs)}")
    
