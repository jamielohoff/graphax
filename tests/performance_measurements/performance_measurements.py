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
    samplesize = 1000
    shape = (1000,)
    
    order = [95, 58, 8, 36, 20, 45, 104, 18, 63, 6, 9, 64, 106, 94, 21, 93, 
            79, 76, 5, 78, 62, 13, 102, 89, 88, 77, 31, 66, 52, 55, 57, 80, 
            90, 87, 60, 51, 27, 25, 92, 112, 39, 29, 33, 75, 47, 68, 103, 50,
            98, 107, 49, 86, 16, 83, 91, 1, 96, 69, 44, 4, 19, 43, 28, 73, 
            108, 81, 10, 7, 37, 54, 105, 110, 70, 22, 3, 26, 34, 35, 99, 72, 
            17, 38, 30, 97, 40, 32, 85, 24, 82, 111, 42, 46, 59, 53, 100, 12,
            109, 14, 74, 15, 56, 41, 48, 0, 2, 71, 11, 23]
    order = [o + 1 for o in order]
    
    mM_order = [6, 7, 9, 10, 14, 18, 19, 21, 22, 53, 63, 64, 65, 67, 77, 78, 
                79, 80, 82, 94, 95, 96, 97, 99, 103, 104, 105, 107, 113, 2, 
                5, 8, 11, 17, 20, 23, 25, 27, 29, 30, 33, 35, 37, 38, 41, 43, 
                45, 46, 49, 50, 54, 55, 58, 59, 69, 71, 73, 74, 81, 84, 86, 
                88, 90, 91, 98, 101, 106, 109, 111, 26, 31, 34, 39, 42, 47, 
                51, 56, 60, 70, 75, 83, 87, 92, 100, 108, 110, 52, 61, 15, 
                28, 36, 44, 48, 57, 72, 1, 89, 13, 112, 3, 32, 40, 4, 16, 76, 
                93, 12, 24]
    
    xs = [jnp.ones(shape)]*6
    gx.plot_performance(RobotArm_6DOF, xs, order, mM_order, "./6DOF_Robot_Arm.png", samplesize=samplesize, caption="6-DOF Robot Arm")
    
    # shapes = [(), (3, 3), (6, 6), (10, 10), (15, 15), (20, 20), (30, 30)]
    # args = [[jnp.ones(shape)]*6 for shape in shapes]
    # gx.plot_performance_over_size(position_angles_6DOF, args, order, "./6DOF_Robot_Arm_sizes.png", samplesize=samplesize)


def measure_RoeFlux():       
    samplesize = 1000
    order = [95, 7, 26, 16, 3, 49, 91, 37, 83, 88, 32, 68, 44, 81, 66, 24, 
            76, 85, 43, 86, 80, 42, 12, 15, 30, 62, 52, 78, 70, 58, 72, 56, 
            39, 94, 47, 10, 90, 46, 99, 1, 25, 41, 28, 71, 36, 57, 31, 21, 
            27, 8, 5, 33, 89, 84, 59, 20, 77, 73, 87, 75, 53, 97, 93, 64, 18, 
            45, 13, 74, 67, 79, 63, 60, 0, 48, 4, 65, 50, 92, 17, 6, 19, 9, 
            69, 55, 61, 82, 51, 40, 14, 35, 54, 38, 22, 2, 23, 11, 34, 29]
    order = [o + 1 for o in order]
    
    mM_order = [4, 5, 8, 9, 16, 17, 25, 27, 31, 33, 38, 43, 44, 45, 69, 84, 1, 2,
                10, 13, 18, 21, 26, 28, 32, 34, 37, 39, 42, 47, 50, 53, 57, 59, 
                62, 64, 66, 67, 68, 71, 73, 75, 76, 77, 80, 81, 83, 85, 86, 87, 
                91, 92, 95, 11, 14, 19, 22, 51, 54, 58, 60, 63, 65, 72, 79, 88, 
                90, 93, 96, 3, 6, 7, 15, 29, 40, 56, 61, 74, 78, 82, 48, 89, 94, 
                23, 35, 46, 24, 70, 41, 98, 100, 12, 20, 30, 49, 52, 55, 36]
    
    shape = (600,)
    xs = [.01, .02, .02, .01, .03, .03]
    xs = [jnp.ones(shape)*x for x in xs]
    gx.plot_performance(RoeFlux_1d, xs, order, mM_order, "./RoeFlux.png", samplesize=samplesize, caption="Roe Flux")
    
    # shapes = [(1,), (5,), (10), (25,), (50,), (100,), (250,), (500,)]# , (15, 15), (20, 20), (30, 30)]
    # args = [[jnp.ones(shape)]*6 for shape in shapes]
    # gx.plot_performance_over_size(RoeFlux_1d, args, "./RoeFlux_sizes.png", samplesize=samplesize)
  
 
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

    
def measure_f():            
    samplesize = 10000
    
    order = [60, 36, 58, 42, 63, 18, 5, 64, 16, 49, 52, 27, 56, 35, 71, 43, 79, 
            57, 26, 54, 17, 12, 48, 69, 30, 41, 67, 22, 33, 55, 8, 40, 75, 24, 
            31, 65, 15, 2, 50, 21, 77, 39, 10, 46, 53, 25, 66, 45, 9, 19, 14, 
            11, 37, 23, 32, 61, 28, 44, 38, 7, 29, 76, 62, 47, 13, 34, 0, 74, 3, 
            51, 1, 72, 59, 4, 6, 20]
    order = [o + 1 for o in order]
    
    mM_order = [43, 41, 38, 36, 35, 37, 49, 14, 22, 24, 28, 32, 42, 47, 50, 53, 
                56, 57, 60, 61, 63, 69, 71, 75, 79, 6, 10, 15, 18, 25, 27, 26, 
                45, 55, 59, 64, 13, 19, 30, 62, 9, 11, 17, 44, 58, 67, 77, 20, 
                31, 34, 40, 1, 8, 33, 39, 48, 72, 76, 46, 66, 4, 7, 54, 29, 51, 
                12, 23, 65, 16, 74, 52, 5, 21, 3, 2]

    a = jnp.ones(4)
    b = jnp.ones((2, 3))
    c = jnp.ones((4, 4))
    d = jnp.ones((4, 1))
    xs = [a, b, c, d]
    
    grad_f = jax.jit(gx.jacve(f, order=order, argnums=(0, 1, 2, 3)))
    print(jax.make_jaxpr(grad_f)(*xs))
    
    gx.plot_performance(f, xs, order, mM_order, "./f.png", samplesize=samplesize)
        

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

