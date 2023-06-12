import os

from graphax.dataset.utils import read
from graphax.dataset.benchmark import make_benchmark_dataset

make_benchmark_dataset("./test_dataset.hdf5")

print(read("./test_dataset.hdf5", [0,1,2,3,4,5,6,7,8,9]))

