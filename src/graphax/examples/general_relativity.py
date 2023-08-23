import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph

a = .5
b = .9
M = 1.

def make_Kerr_Sen_metric():
    def g(t, r, theta, phi):
        sigma = r*(r+2.*b) + a**2 * jnp.cos(theta)**2
        k = r*(r+2*b) - 2.*M*r + a**2
        
        gtt = -(1. - 2*M*r/sigma)
        grr = sigma/k
        gthetatheta = sigma
        gphiphi = (sigma + a**2*jnp.sin(theta)**2 + 2.*M*r*a**2*jnp.sin(theta)**2/sigma)*jnp.sin(theta)**2
        gphit = -2.*M*r*a/sigma*jnp.sin(theta)**2
        return gtt, grr, gthetatheta, gphiphi, gphit

    inputs = [1.]*4
    return make_graph(g, *inputs) 

