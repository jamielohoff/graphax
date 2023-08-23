import time
import timeit

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from graphax.interpreter.to_jaxpr import jacve


def simple():
    def f(x, y):
        z = x * y
        w = z**3
        return w + z, jnp.log(w)

    jax_fwd, jax_rev = [], []
    fwd, rev, cce = [], [], []

    duration = 1000
    shapes = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]

    jac_fwd_f = jax.jit(jacve(f, [1, 2]))
    jac_rev_f = jax.jit(jacve(f, [2, 1]))

    jax_fwd_f = jax.jit(jax.jacfwd(f, argnums=(0, 1)))
    jax_rev_f = jax.jit(jax.jacrev(f, argnums=(0, 1)))

    for shape in shapes:
        print(shape)
        x = jnp.ones(shape)
        y = jnp.ones(shape)


        fwd.append(timeit.timeit(lambda: jac_fwd_f(x, y), number=duration))
        rev.append(timeit.timeit(lambda: jac_rev_f(x, y), number=duration))


        jax_fwd.append(timeit.timeit(lambda: jax_fwd_f(x, y), number=duration))
        jax_rev.append(timeit.timeit(lambda: jax_rev_f(x, y), number=duration))

    plt.figure()
    plt.yscale("log")

    plt.ylabel("Execution time")
    plt.xlabel("Input size")

    plt.plot(jax_fwd, label="Jax forward")
    plt.plot(jax_rev, label="Jax reverse")
    plt.plot(fwd, label="Sparse forward")
    plt.plot(rev, label="Sparse reverse")

    plt.xticks(range(len(shapes)), [str(s) for s in shapes])
    plt.legend()
    plt.savefig("simple.png")
    
    
def Helmholtz():
    
    def Helmholtz(x):
        return x*jnp.log(x / (1. + -jnp.sum(x)))
    
    jax_fwd, jax_rev = [], []
    fwd, rev, cce = [], [], []

    
    jac_fwd_f = jax.jit(jacve(Helmholtz, [1, 2, 3, 4, 5]))
    jac_rev_f = jax.jit(jacve(Helmholtz, [5, 4, 3, 2, 1]))
    jac_cc_f = jax.jit(jacve(Helmholtz, [2, 5, 4, 3, 1]))

    jax_fwd_f = jax.jit(jax.jacfwd(Helmholtz, argnums=(0,)))
    jax_rev_f = jax.jit(jax.jacfwd(Helmholtz, argnums=(0,)))
        
    duration = 10000
    shapes = [(5,), (20,), (50,), (100,), (200,), (500,), (1000, ), (2000,)]

    for shape in shapes:
        print(shape)
        x = jnp.ones(shape)
        x = x/x.size


        cce.append(timeit.timeit(lambda: jac_cc_f(x), number=10000))
        fwd.append(timeit.timeit(lambda: jac_fwd_f(x), number=duration))
        rev.append(timeit.timeit(lambda: jac_rev_f(x), number=duration))


        jax_fwd.append(timeit.timeit(lambda: jax_fwd_f(x), number=duration))
        jax_rev.append(timeit.timeit(lambda: jax_rev_f(x), number=duration))

    plt.figure()
    plt.yscale("log")

    plt.ylabel("Execution time in s")
    plt.xlabel("Input size")

    plt.plot(jax_fwd, label="Jax forward")
    plt.plot(jax_rev, label="Jax reverse")
    plt.plot(fwd, label="Sparse forward")
    plt.plot(rev, label="Sparse reverse")
    plt.plot(cce, label="CCE")

    plt.xticks(range(len(shapes)), [str(s) for s in shapes])
    plt.legend()
    plt.savefig("Helmholtz.png")
    

def Perceptron():
    def Perceptron(x, W1, b1, W2, b2, y):
        y1 = W1 @ x
        z1 = y1 + b1
        a1 = jnp.tanh(z1)
        
        y2 = W2 @ a1
        z2 = y2 + b2
        a2 = jnp.tanh(z2)
        d = a2 - y
        return .5*jnp.sum(d**2)

    key = jrand.PRNGKey(42)
    rev, jax_rev = [], []
    
    jac_rev_f = jax.jit(jacve(Perceptron, order=[9, 8, 7, 6, 5, 4, 3, 2, 1]))
    jax_rev_f = jax.jit(jax.jacrev(Perceptron, argnums=(0, 1, 2, 3, 4, 5)))
      
    duration = 1000
    scales = [1, 2, 5, 10, 20, 50, 100, 200, 500]

    for s in scales:
        x = jnp.ones(4*s)
        y = jrand.normal(key, (4*s,))

        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8*s, 4*s))
        b1 = jrand.normal(b1key, (8*s,))


        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (4*s, 8*s))
        b2 = jrand.normal(b2key, (4*s,))
        
        rev.append(timeit.timeit(lambda: jac_rev_f(x, W1, b1, W2, b2, y), number=duration))
        jax_rev.append(timeit.timeit(lambda: jax_rev_f(x, W1, b1, W2, b2, y), number=duration))
        
        # st = time.time()
        # for _ in range(duration):
        #     key, subkey = jrand.split(key, 2)
        #     x = jrand.normal(subkey, (4*s,))
        #     grad = jac_rev_f(x, W1, b1, W2, b2, y)[1]
        # rev.append(time.time() - st)

        # st = time.time()
        # for _ in range(duration):
        #     key, subkey = jrand.split(key, 2)
        #     x = jrand.normal(subkey, (4*s,))
        #     grad = jax_rev_f(x, W1, b1, W2, b2, y)[1]
        # jax_rev.append(time.time() - st)

    plt.figure()
    plt.yscale("log")

    plt.ylabel("Execution time in s")
    plt.xlabel("Input size")

    plt.plot(jax_rev, label="Jax reverse")
    plt.plot(rev, label="Sparse reverse")

    plt.xticks(range(len(scales)), [str(s) for s in scales])
    plt.legend()
    plt.savefig("Perceptron.png")
    
import graphax as gx  
from graphax.examples.differential_kinematics import position_angles_6DOF, make_6DOF_robot


def RobotArm():
    key = jrand.PRNGKey(42)
    rev, jax_rev = [], []
    
    edges = make_6DOF_robot()
    order = gx.minimal_markowitz(edges)
    
    jac_mM_f = jax.jit(jacve(position_angles_6DOF, order=order, argnums=(0, 1, 2, 3, 4, 5)))
    
    jac_rev_f = jax.jit(jacve(position_angles_6DOF, order="rev", argnums=(0, 1, 2, 3, 4, 5)))
    jax_rev_f = jax.jit(jax.jacrev(position_angles_6DOF, argnums=(0, 1, 2, 3, 4, 5)))
    
    jac_fwd_f = jax.jit(jacve(position_angles_6DOF, order="fwd", argnums=(0, 1, 2, 3, 4, 5)))
    jax_fwd_f = jax.jit(jax.jacfwd(position_angles_6DOF, argnums=(0, 1, 2, 3, 4, 5)))
      
    sample_size = 100
    duration = 100
    
    mM_measurements = []
    fwd_measurements = []
    rev_measurements = []
    jax_fwd_measurements = []
    jax_rev_measurements = []
    
    for s in range(sample_size):   
                
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()     
            
            st = time.time()
            grad = jac_fwd_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            fwd_measurements.append(dt)
        
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jax_fwd_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            jax_fwd_measurements.append(dt)
        
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jac_mM_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            mM_measurements.append(dt)

        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jac_rev_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            rev_measurements.append(dt)

        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jax_rev_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            jax_rev_measurements.append(dt)
    
    mM_measurements = jnp.array(mM_measurements[1:])
    fwd_measurements = jnp.array(fwd_measurements[1:])
    rev_measurements = jnp.array(rev_measurements[1:])
    jax_fwd_measurements = jnp.array(jax_fwd_measurements[1:])
    jax_rev_measurements = jnp.array(jax_rev_measurements[1:])
    
    fwd_mean = jnp.mean(fwd_measurements)
    rev_mean = jnp.mean(rev_measurements)
    mM_mean = jnp.mean(mM_measurements)
    jax_fwd_mean = jnp.mean(jax_fwd_measurements)
    jax_rev_mean = jnp.mean(jax_rev_measurements)
    
    fwd_std = jnp.std(fwd_measurements)
    rev_std = jnp.std(rev_measurements)
    mM_std = jnp.std(mM_measurements)
    jax_fwd_std = jnp.std(jax_fwd_measurements)
    jax_rev_std = jnp.std(jax_rev_measurements)
    
    fig, ax = plt.subplots()
    x_pos = jnp.arange(0, 5)
    modes = ["fwd", "jax fwd", "CC", "rev", "jax rev"]
    runtimes = [fwd_mean, jax_fwd_mean, mM_mean, rev_mean, jax_rev_mean]
    runtime_errors = [fwd_std, jax_fwd_std, mM_std, rev_std, jax_rev_std]
    ax.bar(x_pos, runtimes, yerr=runtime_errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    
    ax.set_ylabel("Evaluation time in [s]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_title("Jacobian evaluation times for different modes")
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig("./6DOF_Robot_Arm.png")


from graphax.examples.roe import make_1d_roe_flux, roe_flux

def RoeFlux():
    key = jrand.PRNGKey(42)
    rev, jax_rev = [], []
    
    edges = make_1d_roe_flux()
    order = gx.minimal_markowitz(edges)
    
    jac_mM_f = jax.jit(jacve(roe_flux, order=order, argnums=(0, 1, 2, 3, 4, 5)))
    
    jac_rev_f = jax.jit(jacve(roe_flux, order="rev", argnums=(0, 1, 2, 3, 4, 5)))
    jax_rev_f = jax.jit(jax.jacrev(roe_flux, argnums=(0, 1, 2, 3, 4, 5)))
    
    jac_fwd_f = jax.jit(jacve(roe_flux, order="fwd", argnums=(0, 1, 2, 3, 4, 5)))
    jax_fwd_f = jax.jit(jax.jacfwd(roe_flux, argnums=(0, 1, 2, 3, 4, 5)))
      
    sample_size = 100
    duration = 100
    
    mM_measurements = []
    fwd_measurements = []
    rev_measurements = []
    jax_fwd_measurements = []
    jax_rev_measurements = []
    
    for s in range(sample_size):   
                
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()     
            
            st = time.time()
            grad = jac_fwd_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            fwd_measurements.append(dt)
        
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jax_fwd_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            jax_fwd_measurements.append(dt)
        
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jac_mM_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            mM_measurements.append(dt)

        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jac_rev_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            rev_measurements.append(dt)

        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            xs = jrand.normal(subkey, (6,)).tolist()
            
            st = time.time()
            grad = jax_rev_f(*xs)
            jax.block_until_ready(grad)
            dt = time.time() - st
            jax_rev_measurements.append(dt)
    
    mM_measurements = jnp.array(mM_measurements[1:])
    fwd_measurements = jnp.array(fwd_measurements[1:])
    rev_measurements = jnp.array(rev_measurements[1:])
    jax_fwd_measurements = jnp.array(jax_fwd_measurements[1:])
    jax_rev_measurements = jnp.array(jax_rev_measurements[1:])
    
    fwd_mean = jnp.mean(fwd_measurements)
    rev_mean = jnp.mean(rev_measurements)
    mM_mean = jnp.mean(mM_measurements)
    jax_fwd_mean = jnp.mean(jax_fwd_measurements)
    jax_rev_mean = jnp.mean(jax_rev_measurements)
    
    fwd_std = jnp.std(fwd_measurements)
    rev_std = jnp.std(rev_measurements)
    mM_std = jnp.std(mM_measurements)
    jax_fwd_std = jnp.std(jax_fwd_measurements)
    jax_rev_std = jnp.std(jax_rev_measurements)
    
    fig, ax = plt.subplots()
    x_pos = jnp.arange(0, 5)
    modes = ["fwd", "jax fwd", "CC", "rev", "jax rev"]
    runtimes = [fwd_mean, jax_fwd_mean, mM_mean, rev_mean, jax_rev_mean]
    runtime_errors = [fwd_std, jax_fwd_std, mM_std, rev_std, jax_rev_std]
    ax.bar(x_pos, runtimes, yerr=runtime_errors, align='center', alpha=0.5, ecolor='black', capsize=10)
    
    ax.set_ylabel("Evaluation time in [s]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_title("Jacobian evaluation times for different modes")
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig("./RoeFlux.png")


# simple()
# Helmholtz()
# Perceptron()
RoeFlux()

