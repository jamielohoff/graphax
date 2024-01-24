from timeit import default_timer as timer

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx


from graphax.examples import Simple, Helmholtz, RobotArm_6DOF, RoeFlux_1d, g, KerrSenn_metric, CloudSchemes_step, f


def measure_simple():
    samplesize = 100
    duration = 1000
    shapes = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    args = [[jnp.ones(shape)*0.01] for shape in shapes]
    gx.plot_performance_over_size(Simple, args, "rev", "./simple.png", samplesize=samplesize, loop_duration=duration)
    

def measure_Helmholtz():
    duration = 10
    samplesize = 100
    
    shapes = [(5,), (20,), (50,), (100,), (200,), (500,), (1000, ), (2000,)]
    args = [[jnp.ones(shape)*0.01] for shape in shapes]
    gx.plot_performance_over_size(Helmholtz, args,  "./Helmholtz.png", samplesize=samplesize, loop_duration=duration)
            

def measure_Perceptron():
    def Perceptron(x, W1, b1, W2, b2, y):
        y1 = jnp.tanh(W1 @ x + b1)
        
        y2 = jnp.tanh(W2 @ y1 + b2)
        d = y2 - y
        return .5*jnp.sum(d**2)
    
    key = jrand.PRNGKey(42)
    rev, jax_rev = [], []
    
    rev_f = jax.jit(gx.jacve(Perceptron, order="rev", argnums=(1, 2, 3, 4)))
    jax_rev_f = jax.jit(jax.jacrev(Perceptron, argnums=(1, 2, 3, 4)))
    
    print(jax.make_jaxpr(rev_f)(jnp.ones(4), jnp.ones((8, 4)), jnp.ones((8,)), jnp.ones((2, 8)), jnp.ones((2,)), jnp.ones(2)))
    print(jax.make_jaxpr(jax_rev_f)(jnp.ones(4), jnp.ones((8, 4)), jnp.ones((8,)), jnp.ones((2, 8)), jnp.ones((2,)), jnp.ones(2)))
      
    samplesize = 1000
    scales = [10, 20, 50, 100, 200, 500, 1000, 2000]
            
    for s in scales:
        x = jnp.ones(4*s)
        y = jrand.normal(key, (2*s,))
        
        w1key, b1key, key = jrand.split(key, 3)
        W1 = jrand.normal(w1key, (8*s, 4*s))
        b1 = jrand.normal(b1key, (8*s,))

        w2key, b2key, key = jrand.split(key, 3)
        W2 = jrand.normal(w2key, (2*s, 8*s))
        b2 = jrand.normal(b2key, (2*s,))
                        
        args = (x, W1, b1, W2, b2, y)
        
        rev_f = jax.jit(gx.jacve(Perceptron, order="rev", argnums=(1, 2, 3, 4)))
        def measure(xs):
            st = timer()
            out = rev_f(*xs)
            # jax.block_until_ready(out)
            dt = timer() - st
            return dt
            
        measurements = jnp.array([measure([arg for arg in args]) for i in range(samplesize)])
        rev.append(measurements[1:].sum())
    
        jax_rev_f = jax.jit(gx.jacve(Perceptron, order="rev", argnums=(1, 2, 3, 4)))
        def measure(xs):
            st = timer()
            out = jax_rev_f(*xs)
            # jax.block_until_ready(out)
            dt = timer() - st
            return dt
            
        measurements = jnp.array([measure([arg for arg in args]) for i in range(samplesize)])
        jax_rev.append(measurements[1:].sum())
        
    print(rev, jax_rev)

    plt.figure()
    plt.yscale("log")

    plt.ylabel("Execution time in s")
    plt.xlabel("Input size")

    plt.plot(rev, label="Sparse reverse")
    plt.plot(jax_rev, label="Jax reverse")

    plt.xticks(range(len(scales)), [str(s) for s in scales])
    plt.legend()
    plt.savefig("Perceptron.png")
            

def measure_RobotArm():    
    samplesize = 100
        
    shape = (1000,)
    
    xs = [jnp.ones(shape)]*6
    gx.plot_performance(RobotArm_6DOF, xs, "./6DOF_Robot_Arm.png", samplesize=samplesize, caption="6-DOF Robot")
    
    # shapes = [(), (3, 3), (6, 6), (10, 10), (15, 15), (20, 20), (30, 30)]
    # args = [[jnp.ones(shape)]*6 for shape in shapes]
    # gx.plot_performance_over_size(position_angles_6DOF, args, order, "./6DOF_Robot_Arm_sizes.png", samplesize=samplesize)


def measure_RoeFlux():       
    samplesize = 10000
        
    shape = (60,) 
    xs = [jnp.zeros(shape)]*6
    gx.plot_performance(RoeFlux_1d, xs, "./RoeFlux.png", samplesize=samplesize, caption="Roe Flux")
    
    shapes = [(1,), (5,), (10), (25,), (50,), (100,), (250,), (500,)]# , (15, 15), (20, 20), (30, 30)]
    args = [[jnp.ones(shape)]*6 for shape in shapes]
    gx.plot_performance_over_size(RoeFlux_1d, args, "./RoeFlux_sizes.png", samplesize=samplesize)
  
 
def measure_g():            
    samplesize = 1000
    shape = (60,)
    xs = [jnp.zeros(shape)*0.01]*15
    gx.plot_performance(g, xs, "./g.png", samplesize=samplesize, caption="an arbitrary function")

    
def measure_KerrSen():    
    samplesize = 1000
    shape = (1000,)
    xs = [jnp.zeros(shape)]*4
    gx.plot_performance(KerrSenn_metric, xs, "./KerrSen.png", samplesize=samplesize)
    
    
def measure_CloudSchemes():            
    samplesize = 1000
    shape = (100,)
    xs = [jnp.zeros(shape)]*11
    gx.plot_performance(CloudSchemes_step, xs, "./CloudSchemes.png", samplesize=samplesize)

    
# NOTE Does not work straight-forwardly since it has reshape operations!
def measure_f():            
    samplesize = 1000
    
    a = jnp.ones(4)
    b = jnp.ones((2, 3))
    c = jnp.ones((4, 4))
    d = jnp.ones((3, 3))
    e = jnp.ones((4, 1))
    xs = [a, b, c, d, e]
    
    grad_f = jax.jit(gx.jacve(f, order="rev", argnums=(0, 1, 2, 3, 4)))
    print(jax.make_jaxpr(grad_f)(*xs))
    
    gx.plot_performance(f, xs, "./f.png", samplesize=samplesize)
    
    
def measure_softmax_attention():
    pass
    

# simple()
# measure_Helmholtz()
# measure_Perceptron()
# measure_RobotArm()
# measure_g()
measure_RoeFlux()
# measure_KerrSen()
# measure_CloudSchemes()
# measure_f()
# measure_SNN()

