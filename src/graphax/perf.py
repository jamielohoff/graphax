from typing import Callable, Sequence, Union
from timeit import default_timer as timer
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp

from chex import Array

from .core import jacve

Order = Union[str, Sequence[int]]


def measure_execution_time(f: Callable, 
                            args: Sequence[Array], 
                            order: Order,
                            samplesize: int = 1000, 
                            print_results: bool = False) -> Sequence[int]:
    """
    TODO docstring
    """
    measurements = []
    argnums = list(range(len(args)))
    
    grad_f = ()
    vmap_f = jax.vmap(f, in_axes=[0]*len(args))
    grad_f = jax.jit(jacve(vmap_f, order=order, argnums=argnums))
    
    def measure(xs):
        st = timer()
        out = grad_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        return dt
        
    measurements = [measure([arg for arg in args])*1000 for i in tqdm(range(samplesize))]
    if print_results:
        print(measurements)
    
    plot = sns.histplot(measurements[1:], bins=50, stat="probability")
    fig = plot.get_figure()
    fig.savefig("./test.png")
    
    del measure
    del grad_f
    
    # Exclude first measurement due to JIT compilation
    return jnp.array(measurements[1:])


def measure_execution_time_with_jax(f: Callable, 
                                    args: Sequence[Array],
                                    samplesize: int = 100) -> Sequence[int]:
    """
    TODO docstring
    """
    fwd_measurements, rev_measurements = [], []
    argnums = list(range(len(args)))
    vmap_f = jax.vmap(f, in_axes=[0]*len(args))
    
    fwd_f = jax.jit(jax.jacfwd(vmap_f, argnums=argnums))
    rev_f = jax.jit(jax.jacrev(vmap_f, argnums=argnums))
    
    # print(grad_f(*[arg for arg in args]))
    def measure(xs):
        st = timer()
        out = fwd_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        return dt
        
    fwd_measurements = [measure([arg for arg in args])*1000 for i in tqdm(range(samplesize))]
        
    def measure(xs):
        st = timer()
        out = rev_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        return dt
        
    rev_measurements = [measure([arg for arg in args])*1000 for i in tqdm(range(samplesize))]
    
    # Exclude first measurement due to JIT compilation
    return jnp.array(fwd_measurements[1:]), jnp.array(rev_measurements[1:])


def plot_performance(f: Callable,
                    args: Sequence[Array],
                    order: Order, 
                    mM_order: Order,
                    fname: str,
                    samplesize: int = 100,
                    quantiles: Array = jnp.array([0.025, 0.975]),
                    caption: str ="different modes") -> None:
    """
    TODO docstring
    """    
    fwd_measurements = measure_execution_time(f, args, "fwd", samplesize=samplesize)
    rev_measurements = measure_execution_time(f, args, "rev", samplesize=samplesize)
    mM_measurements = measure_execution_time(f, args, mM_order, samplesize=samplesize)
    cc_measurements = measure_execution_time(f, args, order, samplesize=samplesize)
    
    fwd_mean = jnp.mean(fwd_measurements)
    rev_mean = jnp.mean(rev_measurements)
    mM_mean = jnp.mean(mM_measurements)
    cc_mean = jnp.mean(cc_measurements)
    
    print(f"fwd mean: {fwd_mean}, rev mean: {rev_mean}, mM_mean: {mM_mean}, cc_mean: {cc_mean}, ")
    
    fwd_std = jnp.std(fwd_measurements) # jnp.quantile(fwd_measurements, quantiles) - fwd_mean
    rev_std = jnp.std(rev_measurements) # jnp.quantile(rev_measurements, quantiles) - rev_mean
    mM_std = jnp.std(mM_measurements) # jnp.quantile(mM_measurements, quantiles) - mM_mean
    cc_std = jnp.std(cc_measurements) # jnp.quantile(cc_measurements, quantiles) - cc_mean
    
    print(f"fwd std: {fwd_std}, rev std: {rev_std}, mM_std: {mM_std}, cc_std: {cc_std}")
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({"font.size": 16})  
    
    modes = ["forward", "reverse", "Markowitz", "AlphaGrad"]
    x_pos = jnp.arange(0, len(modes))
    runtimes = jnp.stack([fwd_mean, rev_mean, mM_mean, cc_mean])
    runtime_errors = jnp.stack([fwd_std, rev_std, mM_std, cc_std])
    ax.bar(x_pos, runtimes, yerr=runtime_errors, align="center", alpha=0.5, 
            ecolor="black", color="#6096f6", capsize=10)
    
    ax.set_ylabel("Evaluation time [ms]", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(f"Evaluation times of {caption}")
    ax.yaxis.grid(True)  
    # ax.set_ylim((0.025, 0.04))
    
    plt.tight_layout()
    # plt.savefig(fname)
    plt.show()
    
    
def plot_performance_and_jax(f: Callable,
                    args: Sequence[Array],
                    order: Order, 
                    mM_order: Order,
                    fname: str,
                    samplesize: int = 100,
                    quantiles: Array = jnp.array([0.025, 0.975]),
                    caption: str ="different modes") -> None:
    """
    TODO docstring
    """    
    fwd_measurements = measure_execution_time(f, args, "fwd", samplesize=samplesize)
    rev_measurements = measure_execution_time(f, args, "rev", samplesize=samplesize)
    mM_measurements = measure_execution_time(f, args, mM_order, samplesize=samplesize)
    cc_measurements = measure_execution_time(f, args, order, samplesize=samplesize)
    
    jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, args, samplesize=samplesize)
    
    fwd_mean = jnp.mean(fwd_measurements)
    rev_mean = jnp.mean(rev_measurements)
    cc_mean = jnp.mean(cc_measurements)
    mM_mean = jnp.mean(mM_measurements)
    
    print(f"fwd mean: {fwd_mean}, rev mean: {rev_mean}, cc_mean: {cc_mean}, mM_mean: {mM_mean}")
    
    jax_fwd_mean = jnp.mean(jax_fwd_measurements)
    jax_rev_mean = jnp.mean(jax_rev_measurements)
    
    fwd_std = jnp.std(fwd_measurements) # jnp.quantile(fwd_measurements, quantiles) - fwd_mean
    rev_std = jnp.std(rev_measurements) # jnp.quantile(rev_measurements, quantiles) - rev_mean
    cc_std = jnp.std(cc_measurements) # jnp.quantile(cc_measurements, quantiles) - cc_mean
    mM_std = jnp.std(mM_measurements) # jnp.quantile(mM_measurements, quantiles) - mM_mean
    
    print(f"fwd std: {fwd_std}, rev std: {rev_std}, cc_std: {cc_std}, mM_std: {mM_std}")
    
    jax_fwd_std = jnp.std(jax_fwd_measurements) # jnp.quantile(jax_fwd_measurements, quantiles) - jax_fwd_mean
    jax_rev_std = jnp.std(jax_rev_measurements) # jnp.quantile(jax_rev_measurements, quantiles) - jax_rev_mean
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({"font.size": 16})  
    
    modes = ["fwd", "JAX fwd", "rev", "JAX rev", "cc", "mM"]
    x_pos = jnp.arange(0, len(modes))
    runtimes = jnp.stack([fwd_mean, jax_fwd_mean, rev_mean, jax_rev_mean, cc_mean, mM_mean])
    runtime_errors = jnp.stack([fwd_std, jax_fwd_std, rev_std, jax_rev_std, cc_std, mM_std])
    ax.bar(x_pos, runtimes, yerr=runtime_errors, align="center", alpha=0.5, 
            ecolor="black", color="#6096f6", capsize=10)
    
    ax.set_ylabel("Evaluation time [ms]", fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(f"Evaluation times of {caption}")
    ax.yaxis.grid(True)  
    # ax.set_ylim((0.025, 0.04))
    
    plt.tight_layout()
    # plt.savefig(fname)
    plt.show()
        
    
def plot_performance_over_size(f: Callable,
                                args: Sequence[Sequence[Array]],
                                # order: Order, 
                                fname: str,
                                samplesize: int = 1000):
    
    cc_means, cc_stds = [], []
    fwd_means, fwd_stds = [], []
    rev_means, rev_stds = [], []
    jax_fwd_means, jax_fwd_stds = [], []
    jax_rev_means, jax_rev_stds = [], []
    
    for arg in args:
        # cc_measurements = measure_execution_time(f, arg, order, samplesize=samplesize)
        fwd_measurements = measure_execution_time(f, arg, "fwd", samplesize=samplesize)
        rev_measurements = measure_execution_time(f, arg, "rev", samplesize=samplesize)
        
        jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, arg, samplesize=samplesize)
                
        # cc_means.append(jnp.mean(cc_measurements))
        fwd_means.append(jnp.mean(fwd_measurements))
        rev_means.append(jnp.mean(rev_measurements))
        jax_fwd_means.append(jnp.mean(jax_fwd_measurements))
        jax_rev_means.append(jnp.mean(jax_rev_measurements))
                
        # cc_stds.append(jnp.std(cc_measurements))
        fwd_stds.append(jnp.std(fwd_measurements))
        rev_stds.append(jnp.std(rev_measurements))
        jax_fwd_stds.append(jnp.std(jax_fwd_measurements))
        jax_rev_stds.append(jnp.std(jax_rev_measurements))
    
    plt.rc("font", family="serif")
    fig, ax = plt.subplots()
    x_pos = jnp.arange(len(args))
    # modes = ["fwd", "jax fwd", "CC", "rev", "jax rev"]
    ax.errorbar(x_pos, jax_fwd_means, yerr=jax_fwd_stds, label="Jax forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, jax_rev_means, yerr=jax_rev_stds, label="Jax reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, fwd_means, yerr=fwd_stds, label="Graphax forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, rev_means, yerr=rev_stds, label="Graphax reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    # ax.errorbar(x_pos, cc_means, yerr=cc_stds, label="Graphax + AlphaGrad", 
    #             fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    
    ax.set_yscale("log")
    ax.set_ylabel("Evaluation time in [s]")
    ax.set_xlabel("Batchsize")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["1", "5", "10", "25", "50", "100", "250", "500"], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title("Roe Flux evaluation times for different modes and batch sizes")
    ax.legend()
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.savefig(fname)

