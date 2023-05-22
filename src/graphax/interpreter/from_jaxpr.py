import string
from typing import Callable, Tuple, Union, Sequence

import jax
import jax.numpy as jnp

from jax._src.core import Var, Jaxpr

import chex

from ..core import GraphInfo, make_empty_edges, make_graph_info


# TODO This has to be reworked at a later stage to also include these operations
# since we actually differentiation through these operations as well!!! 
# AVOID = ["squeeze",
#         "dynamic_slice",
#         "broadcast_in_dim",
#         "concatenate"]


def filter_literals(vars: Sequence[Var]) -> Sequence[Var]:
    return set([var for var in vars if isinstance(var, Var)])


def filter_eqns(jaxpr: Jaxpr):
    """
    Function that filters out all the squeeze and dynamic_slicing operations
    to generate a computational graph.
    """
    # TODO implement a fast sorting algorithm
    # Remark! nodes that directly feed to the output are treated as output nodes which can lead to 
    # the weird case where an equation with output variable e is treated as a output node while f is
    # an intermediate node. Thus one has to be careful about the mapping of the naming
    # of rows and columns to the intermediate variables!
    vo_eqns = []
    for i, eqn in enumerate(jaxpr.eqns):
        for outvar in eqn.outvars:
            if outvar not in jaxpr.jaxpr._outvars:
                continue
            else:
                for _eqn in jaxpr.eqns[i+1:]:
                    if outvar in _eqn.invars:
                        vo_eqns.append(eqn)
    
    # defining the equations like this is BS    
    # vo_eqns = [eqn for eqn in jaxpr.eqns if filter_literals(jaxpr.jaxpr._outvars) & filter_literals(eqn.invars)]
    o_eqns = [eqn for eqn in jaxpr.eqns if filter_literals(jaxpr.jaxpr._outvars) & filter_literals(eqn.outvars)]
    v_eqns = [eqn for eqn in jaxpr.eqns if eqn in vo_eqns or not eqn in o_eqns]
    return v_eqns, vo_eqns, o_eqns
    

# TODO maybe build computational graph representation in numpy and not in JAX?
# TODO this function is a complexity catastrophy that scales with O(n**3)! - Simplify
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
    v_eqns, vo_eqns, o_eqns = filter_eqns(jaxpr)  
            
    num_i = sum([aval.size for aval in jaxpr.in_avals])
    num_o = sum([aval.size for aval in jaxpr.out_avals])
    num_v = len(jaxpr.eqns)
       
    info = make_graph_info([num_i, num_v, num_o])
    edges = make_empty_edges(info)
    
    # Processing input variables
    variables = {}
    counter = 0
    for invar in jaxpr.jaxpr._invars:
        variables[str(invar)] = counter
        counter += invar.aval.size

    i = 0
    # Process intermediate variables
    print([str(eqn.outvars[0]) for eqn in v_eqns])
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            j = 0
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    # TODO optimize the if-conditions here!
                    if str(invar) not in variables:
                        # TODO this needs to be redone for input variables!!!
                        variables[str(invar)] = counter
                        counter += invar.aval.size
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
    for eqn in o_eqns:
        for outvar in eqn.outvars:
            if eqn in vo_eqns:
                for k in range(outvar.aval.size):
                    j = variables[str(outvar)]
                    edges = edges.at[j+k, i+k].set(1.)
                i += outvar.aval.size
                continue
            j = 0
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    # TODO optimize the if-conditions here!
                    if str(invar) not in variables:
                        variables[str(invar)] = counter
                        counter += invar.aval.size
                    # make difference between vectorization and accumulation
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
    print(variables)
    return edges, info

