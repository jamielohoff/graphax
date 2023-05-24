from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def make_simple() -> Tuple[chex.Array, GraphInfo]:
    def simple(x, y):
        z = x + y
        w = jnp.cos(z)
        return w + z, 2.*w

    edges, info = make_graph(simple, 1., 1.)
    return edges, info


def make_lighthouse() -> Tuple[chex.Array, GraphInfo]:
    def lighthouse(x):
        nu = x[0]
        gamma = x[1]
        omega = x[2]
        t = x[3]
        y1 = nu*jnp.tan(omega*t)/(gamma-jnp.tan(omega*t))
        y2 = gamma*y1
        return jnp.array([y1, y2])

    x = jnp.ones(4)
    edges, info = make_graph(lighthouse, x)
    return edges, info


def make_scalar_assignment_tree() -> Tuple[chex.Array, GraphInfo]:
    def scalar_assignment_tree(u):
        return -10*u[1]*jnp.exp(u[2]) + jnp.log(u[0]) - 3*u[2]*(u[1]-1)*jnp.sqrt(u[0])

    x = jnp.ones(3)
    edges, info = make_graph(scalar_assignment_tree, x)
    return edges, info


def make_hole() -> Tuple[chex.Array, GraphInfo]:
    def hole(x, y, z, w):
        a = y * z
        b = a + x
        c = a + w
        
        d = jnp.cos(b)
        e = jnp.exp(c)
        
        f = d - e
        g = d / e
        h = d * e
        return f, g, h

    edges, info = make_graph(hole, 1., 1., 1., 1.)
    return edges, info

