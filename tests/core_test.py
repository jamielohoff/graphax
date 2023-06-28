from graphax import vertex_eliminate, forward, reverse, make_graph
from graphax.examples import make_simple
from graphax.examples import make_Helmholtz


# Test on simple example
edges = make_simple()
print(edges)

edges, fmas = forward(edges)
print(edges)
print(fmas)

edges = make_simple()
edges, fmas = reverse(edges)
print(fmas)


# Test on Helmholtz example
edges = make_Helmholtz()
print(edges)

# Optimal elimination procedure
edges, fmas = vertex_eliminate(2, edges)
print(edges, fmas)

edges, _fmas = vertex_eliminate(5, edges)
fmas += _fmas
print(edges, _fmas)

edges, _fmas = vertex_eliminate(4, edges)
fmas += _fmas
print(edges, _fmas)

edges, _fmas = vertex_eliminate(3, edges)
fmas += _fmas
print(edges, _fmas)

edges, _fmas = vertex_eliminate(1, edges)
fmas += _fmas
print(edges, _fmas)
print("Result:")
print(fmas)

edges = make_Helmholtz()
edges, fmas = forward(edges)
print(fmas)

edges = make_Helmholtz()
edges, fmas = reverse(edges)
print(fmas)

# Test on neural network
# def NeuralNetwork(x, W1, b1, W2, b2, y):
#     y1 = W1 @ x
#     z1 = y1 + b1
#     a1 = jnp.tanh(z1)
    
#     y2 = W2 @ a1
#     z2 = y2 + b2
#     a2 = jnp.tanh(z2)
#     d = a2 - y
#     e = d**2
#     return .5*jnp.sum(e)

# x = jnp.ones(4)
# y = jnp.ones(4)

# W1 = jnp.ones((3, 4))
# b1 = jnp.ones(3)

# W2 = jnp.ones((4, 3))
# b2 = jnp.ones(4)
# print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))
# edges, info, vertex_mask, attn_mask = make_graph(NeuralNetwork, x, W1, b1, W2, b2, y)

# edges, fmas = tensor_vertex_eliminate(1, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(2, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(3, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(4, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(5, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(6, edges, info)
# # print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(7, edges, info)
# print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(8, edges, info)
# print(edges, fmas)

# edges, fmas = tensor_vertex_eliminate(9, edges, info)
# print(edges, fmas)

# edges, fmas = tensor_reverse(edges, info)
# print(fmas)


