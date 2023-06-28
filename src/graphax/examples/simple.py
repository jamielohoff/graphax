import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def make_simple():
    def simple(x, y):
        z = x * y
        w = jnp.sin(z)
        return w + z, jnp.log(w)

    return make_graph(simple, 1., 1.)


def make_lighthouse():
    def lighthouse(nu, gamma, omega, t):
        y1 = nu*jnp.tan(omega*t)/(gamma-jnp.tan(omega*t))
        y2 = gamma*y1
        return y1, y2

    return make_graph(lighthouse, 1., 1., 1., 1.)


def make_scalar_assignment_tree():
    def scalar_assignment_tree(u, v, w):
        return -10*v*jnp.exp(u) + jnp.log(u) - 3*w*(v-1)*jnp.sqrt(u)

    return make_graph(scalar_assignment_tree, 1., 1., 1.)


def make_hole():
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

    return make_graph(hole, 1., 1., 1., 1.)

