import jax
import jax.numpy as jnp

# SNN definition:
surrogate = lambda x: 1. / (1. + 1.*jnp.abs(x))

def SNN_LIF(x, z, u, W, V):
    '''
    Single layer SNN with implicit and explicit recurrence.
    x: layer spike inputs.
    z: previous timestep layer spike outputs.
    v: membrane potential.
    W: implicit recurrence weights.
    V: explicit recurrence weights.
    '''
    beta = 0.95 # seems to work best with this value
    u_next = beta * u + (1. - beta) * (jnp.dot(W, x) + jnp.dot(V, z))
    surr = surrogate(u_next)
    # Trick so that in forward pass we get the heaviside spike output and in
    # backward pass we get the derivative of surrogate only without heaviside.
    z_next = jax.lax.stop_gradient(jnp.heaviside(u_next, 0.) - surr) + surr
    return z_next, u_next


def SNN_sigma_delta(in_, i, e, i_mem, s, z, W, V):
    input_decay = .2
    membrane_decay = .2
    feedback_decay = .2
    threshold_ = .0

    act_ = jnp.dot(W, in_) + jnp.dot(V, z)
    i = i * input_decay + act_
    e = i - s
    i_mem = i_mem * membrane_decay + e
    surr = surrogate(i_mem - threshold_)
    z_out = jax.lax.stop_gradient(jnp.heaviside(i_mem - threshold_, .0) - surr) + surr
    s = s * feedback_decay + z_out
    return i, e, i_mem, s, z_out

