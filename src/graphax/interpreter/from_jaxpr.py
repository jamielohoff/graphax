import string
from typing import Callable, Tuple, Union, Sequence

import jax
import jax.numpy as jnp

from jax._src.core import Var, Jaxpr

import chex

from ..core import GraphInfo, make_empty_edges, make_graph_info
    

# TODO maybe build computational graph representation in numpy and not in JAX?
def make_graph(f_jaxpr: Union[Jaxpr, Callable], 
               *xs: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    Function that creates a computational graph from a pure JAX input function
    or a jaxpr. Works fairly well, but beware of the caveats:
    1.) 
    2.)
    3.)
    """
    jaxpr = jax.make_jaxpr(f_jaxpr)(*xs) if isinstance(f_jaxpr, Callable) else f_jaxpr
    print(jaxpr) 
            
    num_i = sum([aval.size for aval in jaxpr.in_avals])
    num_o = sum([aval.size for aval in jaxpr.out_avals])
    num_v = sum([outvar.aval.size for eqn in jaxpr.eqns for outvar in eqn.outvars])
       
    info = make_graph_info([num_i, num_v, num_o])
    edges = make_empty_edges(info)
    
    # Processing input variables
    variables = {}
    counter = 0
    for invar in jaxpr.jaxpr._invars:
        variables[str(invar)] = counter
        counter += invar.aval.size

    # Process intermediate variables
    i = 0
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            if str(outvar) not in variables:
                variables[str(outvar)] = counter
                counter += outvar.aval.size
            j = 0
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    if outvar.aval.size > 1 and invar.aval.size > 1:
                        # parallel op
                        for k in range(invar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j+k, i+k].set(1.) 
                    elif outvar.aval.size > 1 and invar.aval.size == 1:
                        # vectorized op
                        for k in range(outvar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j, i+k].set(1.)                      
                    else:
                        # accumulation op
                        for k in range(invar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j+k, i].set(1.)  
                    j += invar.aval.size
            i += outvar.aval.size
            
    # Processing output variables
    for outvar in jaxpr.jaxpr._outvars:
        for k in range(outvar.aval.size):
            j = variables[str(outvar)]
            edges = edges.at[j+k, i+k].set(1.)
        i += outvar.aval.size
    return edges, info

