from graphax.examples import make_Helmholtz
from graphax.dataset.utils import sparsify, densify

edges = make_Helmholtz()


header, sparse_edges = sparsify(edges)
print(sparse_edges)

dense_edges = densify(header, sparse_edges)
print(dense_edges)

