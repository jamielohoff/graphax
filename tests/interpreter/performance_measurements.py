import timeit

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx


def measure_simple():
    def simple(x, y):
        z = x * y
        w = z**3
        return w + z, jnp.log(w)

    samplesize = 100
    duration = 1000
    shapes = [(5, 5), (10, 10), (20, 20), (30, 30), (40, 40), (50, 50)]
    args = [[jnp.ones(shape)*0.01] for shape in shapes]
    gx.plot_performance_over_size(simple, args, "rev", "./simple.png", samplesize=samplesize, loop_duration=duration)
    
    
def measure_Helmholtz():
    
    def Helmholtz(x):
        return x*jnp.log(x / (1. + -jnp.sum(x)))
    
    edges = gx.make_graph(Helmholtz, jnp.ones(4)*0.01)
    order = gx.minimal_markowitz(edges)
    
    duration = 10
    samplesize = 100
    
    shapes = [(5,), (20,), (50,), (100,), (200,), (500,), (1000, ), (2000,)]
    args = [[jnp.ones(shape)*0.01] for shape in shapes]
    gx.plot_performance_over_size(Helmholtz, args, order, "./Helmholtz.png", samplesize=samplesize, loop_duration=duration)
            

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
    
    jac_rev_f = jax.jit(gx.jacve(Perceptron, order="rev"))
    jax_rev_f = jax.jit(jax.jacrev(Perceptron, argnums=(0, 1, 2, 3, 4, 5)))
      
    duration = 1000
    scales = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

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
    
    
from graphax.examples.differential_kinematics import position_angles_6DOF, make_6DOF_robot

def measure_RobotArm():    
    edges = make_6DOF_robot()
    order = gx.minimal_markowitz(edges)
    
    _, ops = gx.forward(edges)
    print("forward", ops)
    _, ops = gx.reverse(edges)
    print("reverse", ops)
    _, ops = gx.cross_country(order, edges)
    print("cc", ops)
    
    samplesize = 100
    duration = 100
        
    shape = (20,20)
    
    xs = [jnp.ones(shape)]*6
    gx.plot_performance(position_angles_6DOF, xs, order, "./6DOF_Robot_Arm.png", samplesize=samplesize, loop_duration=duration)
    
    shapes = [(), (3, 3), (6, 6), (10, 10), (15, 15), (20, 20), (30, 30)]
    args = [[jnp.ones(shape)]*6 for shape in shapes]
    gx.plot_performance_over_size(position_angles_6DOF, args, order, "./6DOF_Robot_Arm_sizes.png", samplesize=samplesize, loop_duration=duration)
    
    
from graphax.examples.simple import make_g, g
 
def measure_G():    
    edges = make_g()
    order = gx.minimal_markowitz(edges)
    
    _, ops = gx.forward(edges)
    print("forward", ops)
    _, ops = gx.reverse(edges)
    print("reverse", ops)
    _, ops = gx.cross_country(order, edges)
    print("cc", ops)
    
        
    samplesize = 500
    duration = 1000
    xs = [jnp.ones((20,))]*15
    print(len(jax.make_jaxpr(gx.jacve(g, order="rev", argnums=list(range(15))))(*xs).eqns))
    print(len(jax.make_jaxpr(gx.jacve(g, order=order, argnums=list(range(15))))(*xs).eqns))
    gx.plot_performance(g, xs, order, "./g.png", samplesize=samplesize, loop_duration=duration)
    

# simple()
# measure_Helmholtz()
# Perceptron()
# measure_RobotArm()
measure_G()

