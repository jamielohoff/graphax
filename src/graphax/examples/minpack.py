import jax
import jax.numpy as jnp


# Propane Combustion - Full Formulation
# To be solved with Newton's Method
p = 40.
R = 10.

def PropaneCombustion(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    px11 = jnp.sqrt(p/x11)
    f1 = x1 + x4 - 3.
    f2 = 2*x1 + x2 + x4 + x7 + x8 + x9 + 2*x10 - R
    f3 = 2*x2 + 2*x5 + x6 + x7 - 8.
    f4 = 2*x3 + x9 - 4*R
    f5 = x2*x4 - x1*x5
    f6 = jnp.sqrt(x2*x4) - jnp.sqrt(x1)*px11*x6
    f7 = jnp.sqrt(x1*x2) - jnp.sqrt(x4)*x7*px11
    f8 = x1 - x4*x8*p/x11   
    f9 = x1*jnp.sqrt(x3) - x4*x9*px11
    f10 = x1**2 - x4**2*x10*p/x11
    f11 = x11 - x1 - x2 - x3 - x4 - x5 - x6 - x7 - x8 - x9 - x10
    return f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11


# Human Heart Diple 
# To be solved with Newton's Method
sigma_mx = 1.
sigma_my = 1.
sigma_A = 1.
sigma_B = 1.
sigma_C = 1.
sigma_D = 1.
sigma_E = 1.
sigma_F = 1.

def HumanHeartDipole(x1, x2, x3, x4, x5, x6, x7, x8):
    f1 = x1 + x2 - sigma_mx
    f2 = x3 + x4 - sigma_my
    f3 = x5*x1 + x6*x2 - x7*x3 -x8*x4 - sigma_A
    f4 = x7*x1 + x8*x2 + x5*x3 + x6*x4 - sigma_B
    
    x57 = (x5**2 - x7**2)
    x68 = (x6**2 - x8**2)
    
    f5 = x1*x57 - 2*x3*x5*x7 + x2*x68 - 2*x4*x6*x8 - sigma_C
    f6 = x3*x57 - 2*x1*x5*x7 + x4*x68 - 2*x2*x6*x8 - sigma_D
    
    x557 = x5*(x5**2 - 3*x7**2)
    x775 = x7*(x7**2 - 3*x5**2)
    x668 = x6*(x6**2 - 3*x8**2)
    x886 = x8*(x8**2 - 3*x6**2)
    
    f7 = x1*x557 + x3*x775 + x2*x668 + x4*x886 - sigma_E
    f8 = x3*x557 - x1*x775 + x4*x668 - x2*x886 - sigma_E
    return f1, f2, f3, f4, f5, f6, f7, f8


def Newton(f, x_0, tol=1e-5, max_iter=15):
    """
    A multivariate Newton root-finding routine.
    Adopted from https://jax.quantecon.org/newtons_method.html
    """
    x = x_0
    f_jac = jax.jacobian(f)
    @jax.jit
    def q(x):
        " Updates the current guess. "
        return x - jnp.linalg.solve(f_jac(x), f(x))
    error = tol + 1
    n = 0
    # TODO Replace this with a lax while loop
    while error > tol:
        n += 1
        if(n > max_iter):
            raise Exception('Max iteration reached without convergence')
        y = q(x)
        if jnp.any(jnp.isnan(y)):
            raise Exception('Solution not found with NaN generated')
        error = jnp.linalg.norm(x - y)
        x = y
        print(f'iteration {n}, error = {error}')
    print('\n' + f'Result = {x} \n')
    return x

