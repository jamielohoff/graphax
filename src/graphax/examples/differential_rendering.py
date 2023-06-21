import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def sdf_sphere(x, c, r):
    return jnp.sum(jnp.square(x - c)) - r

def make_sdf_sphere():
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

def make_sdf_sphere_union():
    x = jnp.ones(3)
    edges, info = make_graph(sdf_sphere_union, x)
    return edges, info

def sdf_box(x1, x2, x3, c1, c2, c3, s1, s2, s3):
    da1 = x1 - c1 - s1/2
    da2 = c1 - x2 - s1/2
    a = jnp.maximum(da1, da2)
    
    db1 = x2 - c2 - s2/2
    db2 = c2 - x2 - s2/2
    b = jnp.maximum(db1, db2)
    
    dc1 = x3 - c3 - s3/2
    dc2 = c3 - x3 - s3/2
    c = jnp.maximum(dc1, dc2)
    
    d = a
    d = jnp.maximum(d, b)
    d = jnp.maximum(d, c)
    return d

def make_sdf_box():
    edges, info = make_graph(sdf_box, 1., 1., 1., 1., 1., 1., 1., 1., 1.)
    return edges, info

