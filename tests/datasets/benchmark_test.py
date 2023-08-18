import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax import reverse, forward, cross_country
from graphax.transforms.markowitz import minimal_markowitz
from graphax.dataset.utils import read
from graphax.dataset import make_benchmark_dataset

key = jrand.PRNGKey(42)
make_benchmark_dataset(key, "./benchmark_dataset.hdf5", size=100)

names, data = read("./benchmark_dataset.hdf5", [0, 1, 2, 3, 4, 5, 6, 7])


for name, edges in zip(names, data):
    print(name)
    edges = jnp.array(edges)
    print(forward(edges)[1])
    print(reverse(edges)[1])
    order = minimal_markowitz(edges)
    print("order", len(order), order)
    print(cross_country(order, edges)[1])

