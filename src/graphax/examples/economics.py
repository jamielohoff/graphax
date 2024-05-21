import jax
import jax.numpy as jnp
import jax.scipy.special as jss

from graphax import jacve


def Phi(z):
    return (1. + jss.erf(z/jnp.sqrt(2.)))/2

def Black76(F, K, r, sigma, T):
    d1 = (jnp.log(F/K) + (sigma**2)/2.*T)/(sigma*jnp.sqrt(T))
    d2 = d1 - sigma*jnp.sqrt(T)
    return jnp.exp(-r*T)*(F*Phi(d1) - K*Phi(d2))

def BlackScholes(S, K, r, sigma, T):
    F = jnp.exp(r*T)*S
    return Black76(F, K, r, sigma, T)

# Calculating the Jacobian of this gives the second-order greeks
def BlackScholes_Jacobian(S, K, r, sigma, T):
    return jacve(BlackScholes, order="rev", argnums=(0, 1, 2, 3, 4))(S, K, r, sigma, T)

