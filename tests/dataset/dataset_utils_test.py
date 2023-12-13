import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.dataset import RandomSampler, Graph2File
from graphax.dataset import create, write, read, delete, densify
            

sampler = RandomSampler(storage_shape=[10, 50, 10],
                        min_num_intermediates=5)

key = jrand.PRNGKey(42)
gen = Graph2File(sampler,
                "./",
                fname_prefix="test",
                num_samples=3, 
                batchsize=3,
                storage_shape=[10, 50, 10])
gen.generate(key=key, sampling_shape=[10, 40, 10])

code, header, graph = read("test-10_50_10_3_64008.hdf5", [0,1,2])

print(code[0].decode("utf-8"))
print(header[0])
print(graph[0])
print(len(graph[0]))
print(len(graph[1]))
print(len(graph[2]))
header = jnp.array(header[0])
graph = jnp.array(graph[0])
assembled = densify(header, graph, shape=[10, 50, 10])
print(assembled)
