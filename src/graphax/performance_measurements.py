from typing import Callable, Sequence, Union
from timeit import default_timer as timer

import matplotlib.pyplot as plt

import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from . import jacve

Order = Union[str, Sequence[int]]


def measure_execution_time(f: Callable, 
                            args: Sequence[Array], 
                            order: Order,
                            samplesize: int = 100, 
                            loop_duration: int = 1000,
                            print_results: bool = True) -> Sequence[int]:
    """
    TODO docstring
    """
    measurements = []
    argnums = list(range(len(args)))
    
    grad_f = ()
    grad_f = jax.jit(jacve(f, order=order, argnums=argnums))
    # def loop_fn(carry, x):
    #     xs = [arg*0.01*carry for arg in args]
    #     grad = jacve(f, order=order, argnums=argnums)(*xs)
    #     jax.block_until_ready(grad)
    #     carry *= 1.01
    #     return carry, grad
    
    for i in range(samplesize):
        xs = [arg*1e-5*i for arg in args]
        st = timer()
        out = grad_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        measurements.append(dt)
        if print_results:
            print(dt)
    
    # Exclude first measurement due to JIT compilation
    return jnp.array(measurements[1:])


def measure_execution_time_with_jax(f: Callable, 
                                    args: Sequence[Array],
                                    samplesize: int = 100, 
                                    loop_duration: int = 1000) -> Sequence[int]:
    """
    TODO docstring
    """
    fwd_measurements, rev_measurements = [], []
    argnums = list(range(len(args)))
    
    fwd_f = jax.jit(jax.jacfwd(f, argnums=argnums))
    rev_f = jax.jit(jax.jacfwd(f, argnums=argnums))
    # def loop_fn(carry, x):
    #     xs = [arg*0.01*carry for arg in args]
    #     grad = jax.jacfwd(f, argnums=argnums)(*xs)
    #     jax.block_until_ready(grad)
    #     carry *= 1.01
    #     return carry, grad
    
    for i in range(samplesize):
        xs = [arg*1e-5*i for arg in args]
        st = timer()
        out = fwd_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        fwd_measurements.append(dt)
        
    # def loop_fn(carry, x):
    #     xs = [arg*0.01*carry for arg in args]
    #     grad = jax.jacrev(f, argnums=argnums)(*xs)
    #     jax.block_until_ready(grad)
    #     carry *= 1.01
    #     return carry, grad
    
    for i in range(samplesize):
        xs = [arg*1e-5*i for arg in args]
        st = timer()
        out = rev_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        rev_measurements.append(dt)
    
    # Exclude first measurement due to JIT compilation
    return jnp.array(fwd_measurements[1:]), jnp.array(rev_measurements[1:])


def plot_performance(f: Callable,
                    args: Sequence[Array],
                    order: Order, 
                    fname: str,
                    samplesize: int = 100,
                    loop_duration: int = 1000,
                    quantiles: Array = jnp.array([0.025, 0.975])) -> None:
    """
    TODO docstring
    """
    
    fwd_measurements = measure_execution_time(f, args, "fwd", samplesize=samplesize, loop_duration=loop_duration)
    rev_measurements = measure_execution_time(f, args, "rev", samplesize=samplesize, loop_duration=loop_duration)
    cc_measurements = measure_execution_time(f, args, order, samplesize=samplesize, loop_duration=loop_duration)
    
    # jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, args, samplesize=samplesize, loop_duration=loop_duration)
    jax_fwd_measurements = jnp.zeros(10)
    jax_rev_measurements = jnp.zeros(10)
    
    fwd_mean = jnp.median(fwd_measurements)
    rev_mean = jnp.median(rev_measurements)
    cc_mean = jnp.median(cc_measurements)
    
    print(f"fwd mean: {fwd_mean}, rev mean: {rev_mean}, mM_mean: {cc_mean}")
    
    jax_fwd_mean = jnp.median(jax_fwd_measurements)
    jax_rev_mean = jnp.median(jax_rev_measurements)
    
    fwd_std = jnp.quantile(fwd_measurements, quantiles) - fwd_mean
    rev_std = jnp.quantile(rev_measurements, quantiles) - rev_mean
    cc_std = jnp.quantile(cc_measurements, quantiles) - cc_mean
    
    print(f"fwd std: {fwd_std}, rev std: {rev_std}, mM_std: {cc_std}")
    
    jax_fwd_std = jnp.quantile(jax_fwd_measurements, quantiles) - jax_fwd_mean
    jax_rev_std = jnp.quantile(jax_rev_measurements, quantiles) - jax_rev_mean
    
    fig, ax = plt.subplots()
    x_pos = jnp.arange(0, 5)
    modes = ["fwd", "jax fwd", "CC", "rev", "jax rev"]
    runtimes = jnp.stack([fwd_mean, jax_fwd_mean, cc_mean, rev_mean, jax_rev_mean])
    runtime_errors = jnp.stack([fwd_std, jax_fwd_std, cc_std, rev_std, jax_rev_std]).T*jnp.array([[-1], [1]])
    ax.bar(x_pos, runtimes, yerr=runtime_errors, align="center", alpha=0.5, ecolor="black", capsize=10)
    
    ax.set_ylabel("Evaluation time in [s]")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_title("Jacobian evaluation times for different modes")
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig(fname)
    
    
def plot_performance_over_size(f: Callable,
                                args: Sequence[Sequence[Array]],
                                order: Order, 
                                fname: str,
                                samplesize: int = 100,
                                loop_duration: int = 1000):
    
    cc_means, cc_stds = [], []
    fwd_means, fwd_stds = [], []
    rev_means, rev_stds = [], []
    jax_fwd_means, jax_fwd_stds = [], []
    jax_rev_means, jax_rev_stds = [], []
    
    for arg in args:
        cc_measurements = measure_execution_time(f, arg, order, samplesize=samplesize, loop_duration=loop_duration)
        fwd_measurements = measure_execution_time(f, arg, "fwd", samplesize=samplesize, loop_duration=loop_duration)
        rev_measurements = measure_execution_time(f, arg, "rev", samplesize=samplesize, loop_duration=loop_duration)
        
        jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, arg, samplesize=samplesize, loop_duration=loop_duration)
                
        cc_means.append(jnp.mean(cc_measurements))
        fwd_means.append(jnp.mean(fwd_measurements))
        rev_means.append(jnp.mean(rev_measurements))
        jax_fwd_means.append(jnp.mean(jax_fwd_measurements))
        jax_rev_means.append(jnp.mean(jax_rev_measurements))
                
        cc_stds.append(jnp.std(cc_measurements))
        fwd_stds.append(jnp.std(fwd_measurements))
        rev_stds.append(jnp.std(rev_measurements))
        jax_fwd_stds.append(jnp.std(jax_fwd_measurements))
        jax_rev_stds.append(jnp.std(jax_rev_measurements))
    
    plt.rc('font', family='serif')
    fig, ax = plt.subplots()
    x_pos = jnp.arange(len(args))
    # modes = ["fwd", "jax fwd", "CC", "rev", "jax rev"]
    ax.errorbar(x_pos, jax_fwd_means, yerr=jax_fwd_stds, label="Jax forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, jax_rev_means, yerr=jax_rev_stds, label="Jax reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, fwd_means, yerr=fwd_stds, label="Vertex forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, rev_means, yerr=rev_stds, label="Vertex reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, cc_means, yerr=cc_stds, label="Vertex cross country", 
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    
    ax.set_yscale("log")
    ax.set_ylabel("Evaluation time in [s]")
    ax.set_xticks(x_pos)
    # ax.set_xticklabels(modes)
    ax.set_title("Jacobian evaluation times for different modes and input sizes")
    ax.legend()
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig(fname)

