import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph

qc, qr, qv = 1., 1., 1.

c = 1.
S = 1.
B = 0.

gTw = 1.

def condensation(qc):
    return c*S*qc**0.33333

def accretion(a2, bc, br, qc, qr):
    return a2*qc**bc*qr**br

def autoconversion(a1, gamma, qc):
    return a1*qc**gamma

def evaporation(e1, d1, e2, d2, qr):
    return (e1*qr**d1 + e2*qr**d2)
    

# Taken from https://gmd.copernicus.org/preprints/gmd-2019-140/gmd-2019-140.pdf
def make_cloud_schemes():
    def simulation_step(a1, a2, e1, e2, delta, gamma, bc, br, d1, d2, chi):
        dqc = condensation(qc) - autoconversion(a1, gamma, qc) - accretion(a2, bc, br, qc, qr)
        
        dqr = autoconversion(a1, gamma, qc) + evaporation(e1, d1, e2, d2, qr) + B - delta*qr**chi
        
        dqv = -condensation(qc) - evaporation(e1, d1, e2, d2, qr)
        
        return dqc, dqr, dqv
    
    inputs = [1.]*11
    return make_graph(simulation_step, *inputs)

