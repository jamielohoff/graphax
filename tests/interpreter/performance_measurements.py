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

    jacs = jax.jit(jacve(f, [2, 1]))

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
        
        st = time.time()
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            x = jrand.normal(subkey, (4*s,))
            grad = jac_rev_f(x, W1, b1, W2, b2, y)[1]
        rev.append(time.time() - st)

        st = time.time()
        for _ in range(duration):
            key, subkey = jrand.split(key, 2)
            x = jrand.normal(subkey, (4*s,))
            grad = jax_rev_f(x, W1, b1, W2, b2, y)[1]
        jax_rev.append(time.time() - st)

    plt.figure()
    plt.yscale("log")

    plt.ylabel("Execution time in s")
    plt.xlabel("Input size")

    plt.plot(jax_rev, label="Jax reverse")
    plt.plot(rev, label="Sparse reverse")

    plt.xticks(range(len(scales)), [str(s) for s in scales])
    plt.legend()
    plt.savefig("Perceptron.png")

simple()
Helmholtz()
Perceptron()

