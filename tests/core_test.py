import jax

from graphax import vertex_eliminate, forward, reverse, make_graph
from graphax.examples import make_simple
from graphax.examples import make_Helmholtz
from graphax.examples import make_Perceptron


# Test on simple example
edges = make_simple()
edges, fmas = jax.jit(forward)(edges)
print(fmas, "8")

edges = make_simple()
edges, fmas = jax.jit(reverse)(edges)
print(fmas, "6")


# Test on Helmholtz example
edges = make_Helmholtz()

# Optimal elimination procedure
edges, fmas = jax.jit(vertex_eliminate)(2, edges)
print(fmas, "1")

edges, _fmas = jax.jit(vertex_eliminate)(5, edges)
fmas += _fmas
print(_fmas, "4")

edges, _fmas = jax.jit(vertex_eliminate)(4, edges)
fmas += _fmas
print(_fmas, "8")

edges, _fmas = jax.jit(vertex_eliminate)(3, edges)
fmas += _fmas
print(_fmas, "4")

edges, _fmas = jax.jit(vertex_eliminate)(1, edges)
fmas += _fmas
print(_fmas, "16")
print("Result:")
print(fmas, "33")

edges = make_Helmholtz()
edges, fmas = jax.jit(forward)(edges)
print(fmas, "56")

edges = make_Helmholtz()
edges, fmas = jax.jit(reverse)(edges)
print(fmas, "36")

# Test on neural network

edges = make_Perceptron()

# edges, fmas = vertex_eliminate(1, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(2, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(3, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(4, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(5, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(6, edges)
# # print(edges, fmas)

# edges, fmas = vertex_eliminate(7, edges)
# print(edges, fmas)

# edges, fmas = vertex_eliminate(8, edges)
# print(edges, fmas)

# edges, fmas = vertex_eliminate(9, edges)
# print(edges, fmas)

edges, fmas = reverse(edges)
print(fmas)


