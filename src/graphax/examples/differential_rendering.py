from typing import Tuple

import jax
import jax.numpy as jnp

import chex

from ..core import GraphInfo
from ..interpreter.from_jaxpr import make_graph


def sdf_sphere(x, c, r):
    return jnp.sum(jnp.square(x - c)) - r

def make_sdf_sphere() -> Tuple[chex.Array, GraphInfo]:
    x = jnp.ones(3)
    c = jnp.zeros(3)
    r = 2
    edges, info = make_graph(sdf_sphere, x, c, r)
    return edges, info

def sdf_sphere_union(x):
    c1 = jnp.array([-1.5, 0, 0])
    c2 = jnp.array([1.5, 0, 0])
    s1 = sdf_sphere(x, c1, 2)
    s2 = sdf_sphere(x, c2, 2)
    return jnp.minimum(s1, s2)

def make_sdf_sphere_union() -> Tuple[chex.Array, GraphInfo]:
    x = jnp.ones(3)
    edges, info = make_graph(sdf_sphere_union, x)
    return edges, info

def sdf_box(x, c, s):
    da1 = x[0] - c[0] - s[0]/2
    da2 = c[0] - x[0] - s[0]/2
    a = jnp.maximum(da1, da2)
    
    db1 = x[1] - c[1] - s[1]/2
    db2 = c[1] - x[1] - s[1]/2
    b = jnp.maximum(db1, db2)
    
    dc1 = x[2] - c[2] - s[2]/2
    dc2 = c[2] - x[2] - s[2]/2
    c = jnp.maximum(dc1, dc2)
    
    d = a
    d = jnp.maximum(d, b)
    d = jnp.maximum(d, c)
    return d

def make_sdf_box() -> Tuple[chex.Array, GraphInfo]:
    x = jnp.ones(3)
    c = jnp.zeros(3)
    s = jnp.ones(3)
    edges, info = make_graph(sdf_box, x, c, s)
    return edges, info

