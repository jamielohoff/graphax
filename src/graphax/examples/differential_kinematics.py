import jax
import jax.numpy as jnp


S = lambda a, b: jnp.cos(a)*jnp.sin(b) + jnp.sin(a)*jnp.cos(b)
C = lambda a, b: jnp.cos(a)*jnp.cos(b) - jnp.sin(a)*jnp.sin(b)

# Taken from https://dergipark.org.tr/en/download/article-file/2289226
def RobotArm_6DOF(t1, t2, t3, t4, t5, t6):
    a1 = 175.
    a2 = 890.
    a3 = 50.
    
    d1 = 575.
    d4 = 1035.
    d6 = 185.
    
    c1 = jnp.cos(t1)
    c2 = jnp.cos(t2)
    c4 = jnp.cos(t4)
    c5 = jnp.cos(t5)
    c6 = jnp.cos(t6)
    
    c23 = C(t2, t3)
    
    s1 = jnp.sin(t1)
    s2 = jnp.sin(t2)
    s4 = jnp.sin(t4)
    s5 = jnp.sin(t5)
    s6 = jnp.sin(t6)
    
    s23 = S(t2, t3)
    
    ax = s5*(c1*c23*c4 + s1*s4) + c1*s23*c5
    ay = s5*(s1*c23*c4 - c1*s4) + s1*s23*c5
    az = s23*c4*s5 - c23*c5
    
    nz = c6*(c23*s5 + s23*c4*c5) - s23*s4*s6
    oz = -s6*(c23*s5 + s23*c4*c5) - s23*s4*c6
    
    # Tait-Bryan angles
    z = jnp.arctan2(ay, ax)
    y_ = jnp.arctan2(jnp.sqrt((1. - az**2)), az)
    z__ = jnp.arctan2(oz, -nz)
    
    x1 = d6*(s5*(c1*c23*c4 + s1*s4) + c1*s23*c5)
    x2 = c1*(a1 + a2*c2 + a3*c23 + d4*s23)
    px = x1 + x2
    
    y1 = d6*(s5*(s1*c23*c4 + c1*s4) + s1*s23*c5)
    y2 = s1*(a1 + a2*c2 + a3*c23 + d4*s23)
    py = y1 + y2
    
    pz = a2*s2 + d1 + a3*s23 - d4*c23 + d6*(s23*c4*s5 - c23*c5)
    return px, py, pz, z, y_, z__

