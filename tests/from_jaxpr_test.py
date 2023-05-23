import jax
import jax.numpy as jnp

from graphax.core import reverse_gpu, forward_gpu
from graphax.interpreter.from_jaxpr import make_graph
from graphax.examples import make_Helmholtz, make_scalar_assignment_tree, make_lighthouse

# def f(x, y):
#     z = x + y
#     w = jnp.cos(z)
#     return w + z, 2.*w

# print(make_graph(f, 1., 1.))


# def simple(x):
#     z = x[0] + x[1]
#     w = jnp.cos(z)
#     return jnp.array([w + z, 2.*w])

# x = jnp.ones(2)
# print(make_graph(simple, x))


# def Helmholtz(x):
#     z = jnp.log(x / (1 - jnp.sum(x)))
#     return x * z

# x = jnp.ones(4)
# edges, info = make_graph(Helmholtz, x)

# out, nops = reverse_gpu(edges, info)
# print(out, nops)

# edges, info = make_Helmholtz()
# print(edges, info)


# def scalar_assignment_tree(u):
#     return -10*u[1]*jnp.exp(u[2]) + jnp.log(u[0]) - 3*u[2]*(u[1]-1)*jnp.sqrt(u[0])

# x = jnp.ones(3)
# print(make_graph(scalar_assignment_tree, x))

# edges, info = make_scalar_assignment_tree()
# print(edges, info)


# def lighthouse(x):
#     nu = x[0]
#     gamma = x[1]
#     omega = x[2]
#     t = x[3]
#     y1 = nu*jnp.tan(omega*t)/(gamma-jnp.tan(omega*t))
#     y2 = gamma*y1
#     return jnp.array([y1, y2])

# x = jnp.ones(4)
# print(make_graph(lighthouse, x))

# edges, info = make_lighthouse()
# print(edges, info)


def f(x):
    a = x[0] * x[1]
    b = x[2] - x[3]
    c = jnp.sin(x[0]) + jnp.cos(x[1])
    d = jnp.exp(x[2]) / x[3]
    e = a + b
    f = jnp.sum(x) + c
    g = jnp.log(d)
    h = jnp.sqrt(jnp.abs(e))
    i = jnp.tan(h)
    j = jnp.maximum(x, 0)
    return jnp.array([e, f, g, i])

print(make_graph(f, jnp.ones(4)))

