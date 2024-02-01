from functools import partial
import tqdm

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrand
import jax.tree_util as jtu

import numpy as np

import optax

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


batchsize = 16
num_heads = 6
dk = 128

tiling = (4, 4)
input_shape = (3, 32, 32)
embedding_dim = input_shape[0]*input_shape[1]*input_shape[2]//(tiling[0]*tiling[1])
seq_len = tiling[0]*tiling[1] + 1

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10(root='./data', train=True,
                            download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batchsize,
                        shuffle=True, num_workers=4)

testset = datasets.CIFAR10(root='./data', train=False,
                            download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batchsize,
                        shuffle=False, num_workers=2)


def multihead_softmax_attention(X, WQ, WK, WV, WO):
    q = WQ @ X
    k = WK @ X
    v = WV @ X
    a = q @ k.T
    out = jnn.softmax(a, axis=0) @ v
    return WO @ out 


# Weigths for first self-attention layer
key = jrand.PRNGKey(42)
qkey, kkey, vkey, okey, key = jrand.split(key, 5)
WQ1 = jrand.normal(qkey, (dk*num_heads, embedding_dim))
WK1 = jrand.normal(kkey, (dk*num_heads, embedding_dim))
WV1 = jrand.normal(vkey, (dk*num_heads, embedding_dim))
WO1 = jrand.normal(okey, (embedding_dim, dk*num_heads))

# Weights for second self-attention layer
qkey, kkey, vkey, okey, key = jrand.split(key, 5)
WQ2 = jrand.normal(qkey, (dk*num_heads, embedding_dim))
WK2 = jrand.normal(kkey, (dk*num_heads, embedding_dim))
WV2 = jrand.normal(vkey, (dk*num_heads, embedding_dim))
WO2 = jrand.normal(okey, (embedding_dim, dk*num_heads))


def MLP(X, W1, b1, W2, b2):
    out = jnp.tanh(W1 @ X + b1)
    return jnp.tanh(W2 @ out + b2)


W1key, b1key, W2key, b2key, key = jrand.split(key, 5)
W1 = jrand.normal(W1key, (1048, embedding_dim))
b1 = jrand.normal(b1key, (1048, 1))
W2 = jrand.normal(W2key, (embedding_dim, 1048))
b2 = jrand.normal(b2key, (embedding_dim, 1))

W3key, b3key, W4key, b4key, key = jrand.split(key, 5)
W3 = jrand.normal(W3key, (512, embedding_dim))
b3 = jrand.normal(b3key, (512, 1))
W4 = jrand.normal(W4key, (10, 512))
b4 = jrand.normal(b4key, (10, 1))

CT = jrand.normal(key, (embedding_dim, 1))


### Softmax cross-entropy loss
def softmax_ce_loss(logits, labels):
    return -jnp.sum(labels*jnn.log_softmax(logits, axis=0))


# Transformer model
@partial(jax.vmap, in_axes=(0, 0, None, None, None, None, None, None, None, None, 
                            None, None, None, None, None, None, None, None, None))
def transformer(X, label, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
                W1, b1, W2, b2, W3, b3, W4, b4, CT):
    X = jnp.concatenate((CT, X), axis=1)
    X = multihead_softmax_attention(X, WQ1, WK1, WV1, WO1)
    X = MLP(X, W1, b1, W2, b2)
    X = multihead_softmax_attention(X, WQ2, WK2, WV2, WO2)
    X = MLP(X, W3, b3, W4, b4)
    print(X.shape, label)
    return softmax_ce_loss(X[:, 0], label)


def model(X, labels, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2, 
            W1, b1, W2, b2, W3, b3, W4, b4, CT):
    return transformer(X, labels, WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
                        W1, b1, W2, b2, W3, b3, W4, b4, CT).sum()
    

### Function to subdivide image into tiles for vision transformer
@jax.vmap
def subdivide(img, tiles = (4, 4)):
    img = jnp.moveaxis(img, 0, -1)
    grid = [jnp.hsplit(half, tiles[0]) for half in jnp.vsplit(img, tiles[1])]
    return jnp.array([h.flatten() for g in grid for h in g]).transpose(1, 0)


### Positional encoding
n = 10000
pe = np.zeros((embedding_dim, seq_len-1))
position = np.arange(0, seq_len-1, dtype=np.float32)[None, :]
div_term = np.power(n, jnp.arange(0, embedding_dim, 2) / embedding_dim)[:, None]
pe[0::2, :] = np.sin(position * div_term)
pe[1::2, :] = np.cos(position * div_term)
pe = jnp.array(pe)

@jax.vmap
def positional_encoding(xs):
    return xs + pe[:xs.shape[0], :]


### Training loop
@jax.jit
def train(batch, labels, weights, opt_state):
    labels = jnn.one_hot(labels, 10)
    xs = subdivide(batch)
    xs = positional_encoding(xs)
    argnums = range(2, 19)
    loss = model(xs, labels, *weights)
    grads = jax.jacrev(model, argnums=argnums)(xs, labels, *weights)
    
    updates, opt_state = optim.update(grads, opt_state)
    weights = jtu.tree_map(lambda x, y: x + y, weights, updates)
    
    return loss, weights

### Preparing optimizer and weights
weights = (WQ1, WK1, WV1, WO1, WQ2, WK2, WV2, WO2,
            W1, b1, W2, b2, W3, b3, W4, b4, CT)

optim = optax.adam(1e-3)
opt_state = optim.init(weights)

# Training loop
pbar = tqdm.tqdm(range(100))
for epoch in pbar:
    for (batch, labels) in tqdm.tqdm(trainloader):
        batch = batch.numpy()
        labels = labels.numpy()
        loss, weights = train(batch, labels, weights, opt_state)
        pbar.set_description(f"loss: {loss}")
    
