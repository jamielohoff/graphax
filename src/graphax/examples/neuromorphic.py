import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph


def superspike_surrogate(beta=10.):

    @jax.custom_jvp
    def heaviside_with_super_spike_surrogate(x):
        return jnp.heaviside(x, 1)

    @heaviside_with_super_spike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_super_spike_surrogate(x)
        tangent_out = 1./(beta*jnp.abs(x)+1.) * x_dot
        return primal_out, tangent_out
    
    return heaviside_with_super_spike_surrogate


surrogate = superspike_surrogate()


def lif(U, I, S, a, b, threshold):
        U_next = a*U + (1.-a)*I
        I_next = b*I + (1.-b)*S
        S_next = surrogate(U_next - threshold)
        
        return U_next, I_next, S_next


# From Bellec et al. e-prop paper
def ada_lif(U, a, S, alpha, beta, rho, threshold):
        U_next = alpha*U + S    
        A_th = threshold + beta*a
        S_next = surrogate(U_next - A_th)
        a_next = rho*a - S_next
        
        return U_next, a_next, S_next


# Single SNN forward pass as done in Zenke&Neftci using time-local loss functions (e.g. regression)
def make_lif_SNN():
    def snn(S_in, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, thresh):
        i1 = W1 @ S_in
        U1, a1, s1 = lif(U1, I1, i1, alpha, beta, thresh)
        i2 = W2 @ s1
        U2, a2, s2 = lif(U2, I2, i2, alpha, beta, thresh)
        i3 = W3 @ s2
        U3, a3, s3 = lif(U3, I3, i3, alpha, beta, thresh)
        return .5*(s3 - S_target)**2, U1, U2, U3, a1, a2, a3
    
    S_in = jnp.ones(2)
    S_target = jnp.ones(2)
    
    U1 = jnp.ones(4)
    U2 = jnp.ones(4)
    U3 = jnp.ones(2)
    
    I1 = jnp.ones(4)
    I2 = jnp.ones(4)
    I3 = jnp.ones(2)
    
    W1 = jnp.ones((4, 2))
    W2 = jnp.ones((4, 4))
    W3 = jnp.ones((2, 4))
    
    alpha = .95
    beta = .9
    threshold = 1.
    return make_graph(snn, S_in, S_target, U1, U2, U3, I1, I2, I3, W1, W2, W3, alpha, beta, threshold)


# Single SNN forward pass as done in Zenke&Neftci using time-local loss functions (e.g. regression)
def make_ada_lif_SNN():
    def snn(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh):
        i1 = W1 @ S_in
        U1, a1, s1 = ada_lif(U1, a1, i1, alpha, beta, rho, thresh)
        i2 = W2 @ s1
        U2, a2, s2 = ada_lif(U2, a2, i2, alpha, beta, rho, thresh)
        i3 = W3 @ s2
        U3, a3, s3 = ada_lif(U3, a3, i3, alpha, beta, rho, thresh)
        return .5*(s3 - S_target)**2, U1, U2, U3, a1, a2, a3
    
    S_in = jnp.ones(2)
    S_target = jnp.ones(2)
    
    U1 = jnp.ones(4)
    U2 = jnp.ones(4)
    U3 = jnp.ones(2)
    
    a1 = jnp.ones(4)
    a2 = jnp.ones(4)
    a3 = jnp.ones(2)
    
    W1 = jnp.ones((4, 2))
    W2 = jnp.ones((4, 4))
    W3 = jnp.ones((2, 4))
    
    alpha = .95
    beta = .9
    rho = .9
    threshold = 1.
    return make_graph(snn, S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, threshold)
        
