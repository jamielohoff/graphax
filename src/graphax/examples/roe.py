import jax
import jax.numpy as jnp

from ..interpreter.from_jaxpr import make_graph

gamma = .5
pressure = lambda u0, u1, u2, u3, u4: (gamma-1.)*(u4-.5*(u1**2+u2**2+u3**2)/u0)
enthalpy = lambda u0, u4, p: (u4 + p)/u0


# Flux term of Euler equation in fluid dynamics
def F(u0, u1, u2, u3, u4, p):
    v1 = u1/u0
    v2 = u2/u0
    v3 = u3/u0
    return u1, p+u1*v1, u1*v2, u1*v3, v1*(p + u4)


# 3d Roe flux as defined in paper Roe[1981]
# TODO write this in vectorized form again
def make_3d_roe_flux():
    def roe_flux(ul0, ul1, ul2, ul3, ul4, ur0, ur1, ur2, ur3, ur4):        
        du0 = ur0 - ul0
        du1 = ur1 - ul1
        du2 = ur2 - ul2
        du3 = ur3 - ul3
        du4 = ur4 - ul4
        
        w1 = jnp.sqrt(ul0) + jnp.sqrt(ur0)
        
        vl1 = ul1/ul0
        vl2 = ul2/ul0
        vl3 = ul3/ul0
        pl = pressure(ul0, ul1, ul2, ul3, ul4)
        hl = enthalpy(ul0, ul4, pl)
        
        vr1 = ur1/ur0
        vr2 = ur2/ur0
        vr3 = ur3/ur0
        pr = pressure(ur0, ur1, ur2, ur3, ur4)
        hr = enthalpy(ur0, ur4, pr)
        
        # Arithmetic mean as in Roe's paper (often called Roe averaging)        
        u = (jnp.sqrt(ul0)*vl1 + jnp.sqrt(ur0)*vr1) / w1
        v = (jnp.sqrt(ul0)*vl2 + jnp.sqrt(ur0)*vr2) / w1
        w = (jnp.sqrt(ul0)*vl3 + jnp.sqrt(ur0)*vr3) / w1
        
        h = (jnp.sqrt(ul0)*hl + jnp.sqrt(ur0)*hr) / w1
        
        q2 = u**2 + v**2 + w**2
        a2 = (gamma-1.)*(h-.5*q2)
        a = jnp.sqrt(a2)
        
        # Take the absolute of the eigenvalues
        lp = jnp.abs(u + a) # 5
        l = jnp.abs(u)
        lm = jnp.abs(u - a) # 1
                
        # Here we calculate the coefficients for the approximation
        c3 = ((gamma-1.)/a2 * ((h-q2)*du0+u*du1+v*du2+w*du3-du4))*l
        c2 = (du3/w - du1)*l
        c1 = (du2/v - du0)*l
        k1 = du0 - c3
        k2 = (du1 - u*du0)/a
        c0 = .5*(k1 - k2)*lm
        c4 = .5*(k1 + k2)*lp
        
        FL0, FL1, FL2, FL3, FL4 = F(ul0, ul1, ul2, ul3, ul4, pl)
        FR0, FR1, FR2, FR3, FR4 = F(ur0, ur1, ur2, ur3, ur4, pr)
        
        F0 = FL0 + FR0
        F1 = FL1 + FR1
        F2 = FL2 + FR2
        F3 = FL3 + FR3
        F4 = FL4 + FR4
        
        dF0 = c0 + c3 + c4*lp
        dF1 = c0*(u-a) + c3*u + c4*(u+a)
        dF2 = c0*v + c1*v + c3*v + c4*v
        dF3 = c0*w + c2*w + c3*w + c4*w
        dF4 = c0*(h-u*a) + c1*v**2 + c2*w**2 + c3*.5*q2 + c4*(h+u*a)
        
        phi0 = .5*(F0 - dF0) 
        phi1 = .5*(F1 - dF1) 
        phi2 = .5*(F2 - dF2) 
        phi3 = .5*(F3 - dF3) 
        phi4 = .5*(F4 - dF4) 
        
        return phi0, phi1, phi2, phi3, phi4
    
    ulr = [1.]*10
    return make_graph(roe_flux, *ulr)    


pressure_1d = lambda u0, u1, u2: (gamma-1.)*(u2-.5*u1**2/u0)

# Flux term of Euler equation in fluid dynamics
def F_1d(u0, u1, u2, p):
    v = u1/u0
    return u1, p+u1*v, v*(p + u2)

# 1d Roe flux as defined in paper Roe[1981]
def roe_flux(ul0, ul1, ul2, ur0, ur1, ur2):        
    du0 = ur0 - ul0
    ulr0 = jnp.sqrt(ul0*ur0)
    
    w1 = jnp.sqrt(ul0) + jnp.sqrt(ur0)
    
    vl = ul1/ul0
    pl = pressure_1d(ul0, ul1, ul2,)
    hl = enthalpy(ul0, ul2, pl)
    
    vr = ur1/ur0
    pr = pressure_1d(ur0, ur1, ur2)
    hr = enthalpy(ur0, ur2, pr)
    
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

def make_1d_roe_flux():    
    ulr = [1.]*6
    return make_graph(roe_flux, *ulr)

