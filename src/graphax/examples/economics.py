import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import jax.random as jrand

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
    return jax.jacrev(BlackScholes, argnums=(0, 1, 2, 3, 4))(S, K, r, sigma, T)

# jaxpr = jax.make_jaxpr(BlackScholes_Jacobian)(1., 1., 1., 1., 1.)
# print(len(jaxpr.jaxpr.eqns))

# Jacobian of BlackScholes == Greeks, i.e. internal KPIs of the model
# Risk calculation in finance uses AD, has to be very fast

# # LIBOR market model
# N = 2
# time_step = 1
# maturity = 3
# key = jrand.PRNGKey(42)

# # zero_curve = jnp.array([0.0074,0.0074,0.0077,0.0082,0.0088,0.0094,0.0101,0.0108,0.0116,0.0123,0.0131])
# zero_curve = jnp.array([0.0074,0.0074,0.0077])
# # forward_rate_volatilities_three_factor = jnp.array([[0.1365,0.1928,0.1672,0.1698,0.1485,0.1395,0.1261,0.1290,0.1197,0.1097],
# #                                                     [-0.0662,-0.0702 ,-0.0406,-0.0206,0, 0.0169, 0.0306,0.0470, 0.0581, 0.0666],
# #                                                     [ 0.0319 , 0.0225, 0, -0.0198, -0.0347, -0.0163, 0, 0.0151, 0.0280, 0.0384]])

# forward_rate_volatilities_three_factor = jnp.array([[0.1365, 0.1928, 0.1672],
#                                                     [-0.0662, -0.0702, -0.0406],
#                                                     [ 0.0319, 0.0225, 0]])


# def LIBOR_Market_Model(zero_curve, forward_rate_volatilities):
#     steps = int(maturity/time_step)+1
#     t = jnp.zeros(steps)
#     time = 0
#     p = forward_rate_volatilities.shape[0]
#     for i in range(steps):
#         t.at[i].set(time)
#         time += time_step
#     Delta = jnp.full((steps-1),time_step)  
#     # B_0 = jnp.zeros(steps)
#     # can be parallelized
#     # for i in range(steps):
#     #     B_0.at[i].set(1/(1+zero_curve[i])**(i+1))
#     ii = jnp.arange(1, steps)
#     B_0 = 1/(1+zero_curve)**(ii+1)
        
#     # forward_rate_from_zero = jnp.zeros((steps-1,steps-1))
#     # can be parallelized
#     # for i in range(steps-1):
#     #     forward_rate_from_zero.at[i, 0].set(1/Delta[i]*(B_0[i]/B_0[i+1]-1))
    
#     B_0p1 = jnp.roll(B_0, -1)
#     forward_rate_from_zero = 1/Delta[i]*(B_0/B_0p1-1)
        
#     forward_rate_mc = 0
#     for n in range(N):
#         forward_rate = jnp.zeros((steps-1,steps-1))
#         # can be parallelized
#         # for i in range(steps-1):
#         #     forward_rate.at[i, 0].set(forward_rate_from_zero[i][0])
#         forward_rate.at[:, 0].set(forward_rate_from_zero)
#         # TODO This loop can also be parallelized
#         for k in range(1,steps-1):
#             for j in range(k):
#                 sum1 = 0
#                 for i in range(j+1,k+1):
#                     sum2, sum3, sum4 = 0, 0, 0
#                     for q in range(p):
#                         e = jrand.normal(key, ())
#                         sum2 += (forward_rate_volatilities[q][i-j-1]*forward_rate_volatilities[q][k-j-1])
#                         sum3 += forward_rate_volatilities[q][k-j-1]**2
#                         sum4 += forward_rate_volatilities[q][k-j-1]*e*jnp.sqrt(Delta[j])
#                     sum1 += (Delta[i]*forward_rate[i][j]*sum2)/(1+Delta[i]*forward_rate[i][j])
#                 forward_rate.at[k, j+1].set(forward_rate[k][j]*jnp.exp((sum1-sum3/2)*Delta[j]+sum4))
#         forward_rate_mc += forward_rate
#     forward_rate_mc = forward_rate_mc / N
#     return forward_rate_mc

# # print(jax.make_jaxpr(LIBOR_Market_Model)(zero_curve, forward_rate_volatilities_three_factor))


