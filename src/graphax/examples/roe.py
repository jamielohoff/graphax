import jax
import jax.numpy as jnp


gamma = .5
pressure_1d = lambda u0, u1, u2: (gamma-1.)*(u2-.5*u1**2/u0)
enthalpy_1d = lambda u0, u4, p: (u4 + p)/u0


# Flux term of Euler equation in fluid dynamics
def F_1d(u0, u1, u2, p):
    v = u1/u0
    return u1, p+u1*v, v*(p + u2)


# 1d Roe flux as defined in paper Roe[1981]
# Init with xs = [.01, .02, .02, .01, .03, .03]
def RoeFlux_1d(ul0, ul1, ul2, ur0, ur1, ur2):        
    du0 = ur0 - ul0
    ulr0 = jnp.sqrt(ul0*ur0)
    
    w1 = jnp.sqrt(ul0) + jnp.sqrt(ur0)
    
    vl = ul1/ul0
    pl = pressure_1d(ul0, ul1, ul2,)
    hl = enthalpy_1d(ul0, ul2, pl)
    
    vr = ur1/ur0
    pr = pressure_1d(ur0, ur1, ur2)
    hr = enthalpy_1d(ur0, ur2, pr)
    
    dp = pr - pl
    dv = vr - vl
            
    # Arithmetic mean as in Roe's paper (often called Roe averaging)        
    u = (jnp.sqrt(ul0)*vl + jnp.sqrt(ur0)*vr) / w1
    h = (jnp.sqrt(ul0)*hl + jnp.sqrt(ur0)*hr) / w1
    
    q2 = u**2
    a2 = (gamma-1.)*(h-.5*q2)
    a = jnp.sqrt(a2)
    
    lp = jnp.abs(u + a)
    l = jnp.abs(u)
    ln = jnp.abs(u)
            
    # Here we calculate the coefficients for the approximation
    n = a*ulr0
    c0 = (du0 - dp/a2)*l
    c1 = (dv + dp/n)*lp
    c2 = (dv - dp/n)*ln

    FL0, FL1, FL2 = F_1d(ul0, ul1, ul2, pl)
    FR0, FR1, FR2 = F_1d(ur0, ur1, ur2, pr)
    
    F0 = FL0 + FR0
    F1 = FL1 + FR1
    F2 = FL2 + FR2
    
    c = .5*ulr0/a
    
    dF0 = c0 + c*c1 - c*c2
    dF1 = c0*u + c*c1*(u+a) - c*c2*(u-a)
    dF2 = c0*.5*q2 + c*c1*(h+u*a) - c*c2*(h-u*a)
    
    phi0 = .5*(F0 - dF0) 
    phi1 = .5*(F1 - dF1) 
    phi2 = .5*(F2 - dF2) 

    return phi0, phi1, phi2


pressure_3d = lambda u0, u, u4: (gamma-1.)*(u4-.5*jnp.sum(u**2)/u0)
enthalpy_3d = lambda u0, u4, p: (u4 + p)/u0


# Flux term of Euler equation in fluid dynamics
def F_3d(u0, u, v, u4, p):
    v1 = v[0:1]
    _p = jnp.concatenate([p, jnp.zeros(2, dtype=jnp.float32)])
    return u[0:1], _p+u*v, v1*(p + u4)


# 3d Roe flux as defined in paper Roe[1981]
# TODO write this in vectorized form again!
def RoeFlux_3d(ul0, ul, ul4, ur0, ur, ur4):
    ul = ul*1.0
    ur = ur*1.0
    
    du0 = ur0 - ul0
    du = ur - ul
    du1 = du[0:1]
    du2 = du[1:2]
    du3 = du[2:]
    du4 = ur4 - ul4
    
    vl = ul/ul0
    vr = ur/ur0
    w1 = jnp.sqrt(ul0) + jnp.sqrt(ur0)

    # Arithmetic mean as in Roe's paper (often called Roe averaging)        
    uvw = (jnp.sqrt(ul0)*vl + jnp.sqrt(ur0)*vr) / w1
    u = uvw[0:1]
    v = uvw[1:2]
    w = uvw[2:]
    
    pl = pressure_3d(ul0, ul, ul4)
    hl = enthalpy_3d(ul0, ul4, pl)
    
    pr = pressure_3d(ur0, ur, ur4)
    hr = enthalpy_3d(ur0, ur4, pr)
        
    h = (jnp.sqrt(ul0)*hl + jnp.sqrt(ur0)*hr) / w1
    
    q2 = jnp.sum(uvw**2)
    a2 = (gamma-1.)*(h-.5*q2)
    a = jnp.sqrt(a2)
    
    # Take the absolute of the eigenvalues
    lp = u + a # 5
    l = u
    lm = u - a # 1
            
    # Here we calculate the coefficients for the approximation
    c3 = ((gamma-1.)/a2 * ((h-q2)*du0+jnp.dot(uvw, du)-du4))*l
    c2 = (du3/w - du1)*l
    c1 = (du2/v - du0)*l
    k1 = du0 - c3
    k2 = (du1 - u*du0)/a
    c0 = .5*(k1 - k2)*lm
    c4 = .5*(k1 + k2)*lp
    
    FL0, FL, FL4 = F_3d(ul0, ul, vl, ul4, pl)
    FR0, FR, FR4 = F_3d(ur0, ur, vr, ur4, pr)
    
    F0 = FL0 + FR0
    F = FL + FR
    F4 = FL4 + FR4
    
    dF0 = c0 + c3 + c4*lp
    dF1 = c0*lm + c3*u + c4*lp
    dF2 = c0*v + c1*v + c3*v + c4*v
    dF3 = c0*w + c2*w + c3*w + c4*w
    dF4 = c0*(h-u*a) + c1*v**2 + c2*w**2 + c3*.5*q2 + c4*(h+u*a)
    
    dF = jnp.concatenate([dF1, dF2, dF3])
    
    phi0 = .5*(F0 - dF0) 
    phi = .5*(F - dF) 
    phi4 = .5*(F4 - dF4) 
    
    return phi0, phi, phi4
      
