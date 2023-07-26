import jax
import jax.random as jrand

from graphax.dataset.utils import read
from graphax.dataset.benchmark import make_task_dataset

key = jrand.PRNGKey(42)
make_task_dataset(key, "./test_dataset.hdf5")

print(read("./test_dataset.hdf5", [0,1,2,3,4,5,6,7]))

