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


def measure(f: Callable,
            args: Sequence[Array],
            samplesize: int = 1000,
            use_vmap: bool = True,
            quantiles: Array = jnp.array([0.025, 0.975])) -> Sequence[int]:
    measurements = []
    vmap_f = jax.vmap(f, in_axes=[0]*len(args)) if use_vmap is True else f
    jit_f = jax.jit(vmap_f)
    
    def _measure(xs):
        st = timer()
        out = jit_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        return dt
        
    measurements = [_measure(args)*1000 for i in tqdm(range(samplesize))]
    
    del _measure
    del jit_f
    
    # Exclude first few measurements due to JIT compilation
    measurements = jnp.array(measurements[10:])
    median = jnp.median(measurements)
    err = jnp.quantile(measurements, quantiles) - median
    return jnp.array(measurements[10:]), median, err


def measure_execution_time(f: Callable, 
                            args: Sequence[Array], 
                            order: Order,
                            samplesize: int = 1000, 
                            print_results: bool = False,
                            use_vmap: bool = True) -> Sequence[int]:
    """
    TODO docstring
    """
    measurements = []
    argnums = list(range(len(args)))
    
    vmap_f = jax.vmap(f, in_axes=[0]*len(args)) if use_vmap is True else f
    grad_f = jax.jit(jacve(vmap_f, order=order, argnums=argnums))
    # grad_f = jax.jit(jax.vmap(jacve(f, order=order, argnums=argnums), in_axes=[0]*len(args)))

    def measure(xs):
        st = timer()
        out = grad_f(*xs)
        jax.block_until_ready(out)
        dt = timer() - st
        return dt
        
    measurements = [measure([arg for arg in args])*1000 for i in tqdm(range(samplesize))]
    if print_results:
        print(measurements)
        
    del measure
    del grad_f
    
    # Exclude first measurements due to JIT compilation
    return jnp.array(measurements[10:])


def measure_execution_time_with_jax(f: Callable, 
                                    args: Sequence[Array],
                                    samplesize: int = 1000) -> Sequence[int]:
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
    return jnp.array(fwd_measurements[10:]), jnp.array(rev_measurements[10:])


def plot_performance(f: Callable,
                    args: Sequence[Array],
                    order: Order, 
                    mM_order: Order,
                    fname: str,
                    samplesize: int = 1000,
                    quantiles: Array = jnp.array([0.025, 0.975]),
                    caption: str ="different modes",
                    use_vmap: bool = True) -> None:
    """
    TODO docstring
    """    
    cc_measurements = measure_execution_time(f, args, order, samplesize=samplesize, use_vmap=use_vmap)
    mM_measurements = measure_execution_time(f, args, mM_order, samplesize=samplesize, use_vmap=use_vmap)
    rev_measurements = measure_execution_time(f, args, "rev", samplesize=samplesize, use_vmap=use_vmap)
    fwd_measurements = measure_execution_time(f, args, "fwd", samplesize=samplesize, use_vmap=use_vmap)
    
    fwd_med = jnp.median(fwd_measurements)
    rev_med = jnp.median(rev_measurements)
    mM_med = jnp.median(mM_measurements)
    cc_med = jnp.median(cc_measurements)
    
    print(f"fwd median: {fwd_med}, rev median: {rev_med}, mM median: {mM_med}, cc median: {cc_med}")
    
    fwd_err = jnp.quantile(fwd_measurements, quantiles) - fwd_med
    rev_err = jnp.quantile(rev_measurements, quantiles) - rev_med
    mM_err = jnp.quantile(mM_measurements, quantiles) - mM_med
    cc_err = jnp.quantile(cc_measurements, quantiles) - cc_med
    
    print(f"fwd err: {fwd_err}, rev err: {rev_err}, mM err: {mM_err}, cc err: {cc_err}")
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({"font.size": 14})  
    
    modes = ["forward", "reverse", "Markowitz", "AlphaGrad"]
    x_pos = jnp.arange(0, len(modes))
    runtimes = jnp.stack([fwd_med, rev_med, mM_med, cc_med])
    runtime_errors = jnp.stack([fwd_err, rev_err, mM_err, cc_err], axis=1)*jnp.array([[-1.], [1.]])

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
    
    fwd_med = jnp.median(fwd_measurements)
    rev_med = jnp.median(rev_measurements)
    cc_med = jnp.median(cc_measurements)
    mM_med = jnp.median(mM_measurements)
    
    print(f"fwd median: {fwd_med}, rev median: {rev_med}, cc median: {cc_med}, mM median: {mM_med}")
    
    jax_fwd_med = jnp.median(jax_fwd_measurements)
    jax_rev_med = jnp.median(jax_rev_measurements)
    
    fwd_err = jnp.quantile(fwd_measurements, quantiles) - fwd_med
    rev_err = jnp.quantile(rev_measurements, quantiles) - rev_med
    cc_err = jnp.quantile(cc_measurements, quantiles) - cc_med
    mM_err = jnp.quantile(mM_measurements, quantiles) - mM_med
    
    print(f"fwd err: {fwd_err}, rev err: {rev_err}, cc err: {cc_err}, mM err: {mM_err}")
    
    jax_fwd_err = jnp.quantile(jax_fwd_measurements, quantiles) - jax_fwd_med
    jax_rev_err = jnp.quantile(jax_rev_measurements, quantiles) - jax_rev_med
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({"font.size": 16})  
    
    modes = ["fwd", "JAX fwd", "rev", "JAX rev", "cc", "mM"]
    x_pos = jnp.arange(0, len(modes))
    runtimes = jnp.stack([fwd_med, jax_fwd_med, rev_med, jax_rev_med, cc_med, mM_med])
    runtime_errors = jnp.stack([fwd_err, jax_fwd_err, rev_err, jax_rev_err, cc_err, mM_err])
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
    

def plot_performance_jax_only(f: Callable,
                    args: Sequence[Array],
                    name: str,
                    samplesize: int = 1000,
                    quantiles: Array = jnp.array([0.025, 0.975]),
                    caption: str ="different modes") -> None:
    """
    TODO docstring
    """        
    jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, args, samplesize=samplesize)
            
    fwd_med = jnp.median(jax_fwd_measurements)
    rev_med = jnp.median(jax_rev_measurements)
    
    print(f"fwd median: {fwd_med}, rev median: {rev_med}")
        
    fwd_err = jnp.quantile(jax_fwd_measurements, quantiles) - fwd_med
    rev_err = jnp.quantile(jax_rev_measurements, quantiles) - rev_med
    
    print(f"fwd err: {fwd_err}, rev err: {rev_err}")
    
    fig, ax = plt.subplots()
    
    plt.rcParams.update({"font.size": 15})  
    
    modes = ["JAX fwd", "JAX rev"]
    x_pos = jnp.arange(0, len(modes))
    runtimes = jnp.stack([fwd_med, rev_med])
    runtime_errors = jnp.stack([fwd_err, rev_err])
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
                                args: Sequence[Array],
                                order: Order, 
                                mM_order: Order,
                                task: str,
                                ticks: Sequence[int] = (2, 4, 8),
                                samplesize: int = 1000,
                                quantiles = jnp.array([0.025, 0.975])):
    
    cc_medians, cc_errs = [], []
    mM_medians, mM_errs = [], []
    fwd_medians, fwd_errs = [], []
    rev_medians, rev_errs = [], []
    
    for arg in args:
        cc_measurements = measure_execution_time(f, arg, order, samplesize=samplesize)
        mM_order_measurements = measure_execution_time(f, arg, mM_order, samplesize=samplesize)
        fwd_measurements = measure_execution_time(f, arg, "fwd", samplesize=samplesize)
        rev_measurements = measure_execution_time(f, arg, "rev", samplesize=samplesize)
               
        cc_median = jnp.median(cc_measurements)
        mM_median = jnp.median(mM_order_measurements)
        fwd_median = jnp.median(fwd_measurements)
        rev_median = jnp.median(rev_measurements)
        
        cc_medians.append(cc_median)
        mM_medians.append(mM_median)
        fwd_medians.append(fwd_median)
        rev_medians.append(rev_median)
                
        cc_errs.append(jnp.quantile(cc_measurements, quantiles) - cc_median)
        mM_errs.append(jnp.quantile(mM_order_measurements, quantiles) - mM_median)
        fwd_errs.append(jnp.quantile(fwd_measurements, quantiles) - fwd_median)
        rev_errs.append(jnp.quantile(rev_measurements, quantiles) - rev_median)
        
    _arr = jnp.array([[-1.], [1.]])
    cc_errs = jnp.stack(cc_errs, axis=1)*_arr
    mM_errs = jnp.stack(mM_errs, axis=1)*_arr
    fwd_errs = jnp.stack(fwd_errs, axis=1)*_arr
    rev_errs = jnp.stack(rev_errs, axis=1)*_arr
    
    cc_medians = jnp.array(cc_medians)
    mM_medians = jnp.array(mM_medians)
    fwd_medians = jnp.array(fwd_medians)
    rev_medians = jnp.array(rev_medians)
    
    plt.rc("font", family="serif")
    fig, ax = plt.subplots()
    x_pos = jnp.arange(len(args))
    ax.errorbar(x_pos, cc_medians, yerr=cc_errs, label="Graphax + AlphaGrad", 
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, mM_medians, yerr=mM_errs, label="Graphax Markowitz", 
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, fwd_medians, yerr=fwd_errs, label="Graphax forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, rev_medians, yerr=rev_errs, label="Graphax reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    
    ax.set_yscale("log")
    ax.set_ylabel("Evaluation time in [ms]")
    ax.set_xlabel("Batchsize")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in ticks], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(f"{task} evaluation times for different modes and batch sizes")
    ax.legend()
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_performance_over_size_jax(f: Callable,
                                    args: Sequence[Array],
                                    order: Order, 
                                    task: str,
                                    ticks: Sequence[int] = (2, 4, 8),
                                    samplesize: int = 1000,
                                    quantiles = jnp.array([0.025, 0.975])):
    
    cc_medians, cc_errs = [], []
    jax_fwd_medians, jax_fwd_errs = [], []
    jax_rev_medians, jax_rev_errs = [], []
    
    for arg in args:
        cc_measurements = measure_execution_time(f, arg, order, samplesize=samplesize)
        
        jax_fwd_measurements, jax_rev_measurements = measure_execution_time_with_jax(f, arg, samplesize=samplesize)
                
        cc_median = jnp.median(cc_measurements)
        jax_fwd_median = jnp.median(jax_fwd_measurements)
        jax_rev_median = jnp.median(jax_rev_measurements)
        
        cc_medians.append(cc_median)
        jax_fwd_medians.append(jax_fwd_median)
        jax_rev_medians.append(jax_rev_median)
                
        cc_errs.append(jnp.quantile(cc_measurements, quantiles) - cc_median)
        jax_fwd_errs.append(jnp.quantile(jax_fwd_measurements, quantiles) - jax_fwd_median)
        jax_rev_errs.append(jnp.quantile(jax_rev_measurements, quantiles) - jax_rev_median)
        
    _arr = jnp.array([[-1.], [1.]])
    jax_fwd_errs = jnp.stack(jax_fwd_errs, axis=1)*_arr
    jax_rev_errs = jnp.stack(jax_rev_errs, axis=1)*_arr
    cc_errs = jnp.stack(cc_errs, axis=1)*_arr
    
    jax_fwd_medians = jnp.array(jax_fwd_medians)
    jax_rev_medians = jnp.array(jax_rev_medians)
    cc_medians = jnp.array(cc_medians)
    
    font = {"family" : "normal",
            "weight" : "normal",
            "size"   : 15}

    plt.rc("font", **font)
    fig, ax = plt.subplots()
    x_pos = jnp.arange(len(args))
    ax.errorbar(x_pos, jax_fwd_medians, yerr=jax_fwd_errs, label="JAX forward mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    ax.errorbar(x_pos, jax_rev_medians, yerr=jax_rev_errs, label="JAX reverse mode",
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    
    ax.errorbar(x_pos, cc_medians, yerr=cc_errs, label="Graphax + AlphaGrad", 
                fmt='.-', ecolor="black", elinewidth=1, capsize=3)
    
    ax.set_yscale("log")
    ax.set_ylabel("Evaluation time in [ms]")
    ax.set_xlabel("Batchsize")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in ticks], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(f"{task} evaluation times for different modes and batch sizes")
    ax.legend()
    ax.yaxis.grid(True)
    
    plt.tight_layout()
    plt.show()

