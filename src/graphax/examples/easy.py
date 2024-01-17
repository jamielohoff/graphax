import jax
import jax.numpy as jnp
import jax.scipy.special as jss


### Examples from "Evaluting Derivatives" book by Andreas Griewank and Andrea Walther

def Simple(x, y):
    z = x * y
    w = jnp.sin(z)
    return w + z, jnp.log(w)

def Lighthouse(nu, gamma, omega, t):
    y1 = nu*jnp.tan(omega*t)/(gamma-jnp.tan(omega*t))
    y2 = gamma*y1
    return y1, y2

def Hole(x, y, z, w):
    a = y * z
    b = a + x
    c = a + w
    
    d = jnp.cos(b)
    e = jnp.exp(c)
    
    f = d - e
    g = d / e
    h = d * e
    return f, g, h


### General Relativity

a = .5
b = .9
M = 1.

def KerrSenn_metric(t, r, theta, phi):
	sintheta2 = jnp.sin(theta)**2
	sigma = r**2 + 2.*b*r + a**2 * jnp.cos(theta)**2
	k = r**2 + 2*b*r - 2.*M*r + a**2

	gtt = -(1. - 2.*M*r/sigma)
	grr = sigma/k
	gthetatheta = sigma
	gphiphi = (sigma + a**2*sintheta2 + \
				2.*M*r*a**2*sintheta2/sigma)*sintheta2
	gphit = -2.*M*r*a/sintheta2
	return gtt, grr, gthetatheta, gphiphi, gphit


### Thermodynamics and Statistical Mechanics

def Helmholtz(x):
    z = jnp.log(x / (1 + -jnp.sum(x)))
    return x * z

def FreeEnergy(x):
    z = jnp.log(x / (1 - jnp.sum(x)))
    return jnp.sum(x * z)

### Economics

def Phi(z):
    return (1. + jss.erf(z/jnp.sqrt(2.)))/2

def Black76(F, K, r, sigma, T):
    d1 = (jnp.log(F/K) + (sigma**2)/2.*T)/(sigma*jnp.sqrt(T))
    d2 = d1 - sigma*jnp.sqrt(T)
    return jnp.exp(-r*T)*(F*Phi(d1) - K*Phi(d2))

def BlackScholes(S, K, r, sigma, T):
    F = jnp.exp(r*T)*S
    return Black76(F, K, r, sigma, T)


### Meterology

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
def CloudSchemes_step(a1, a2, e1, e2, delta, gamma, bc, br, d1, d2, chi):
    dqc = condensation(qc) - autoconversion(a1, gamma, qc) - accretion(a2, bc, br, qc, qr)
    
    dqr = autoconversion(a1, gamma, qc) + evaporation(e1, d1, e2, d2, qr) + B - delta*qr**chi
    
    dqv = -condensation(qc) - evaporation(e1, d1, e2, d2, qr)
    
    return dqc, dqr, dqv

