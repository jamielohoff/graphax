import string
from typing import Callable, Tuple, Union

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


def filter_literals(vars):
    return set([var for var in vars if isinstance(var, Var)])


def filter_eqns(f_jaxpr):
    """
    Function that filters out all the squeeze and dynamic_slicing operations
    to generate a computational graph.
    """
    # TODO implement a fast sorting algorithm
    # Remark! nodes that directly feed to the output are treated as output nodes which can lead to 
    # the weird case where an equation with output variable e is treated as a output node while f is
    # an intermediate node. Thus one has to be careful about the mapping of the naming
    # of rows and columns to the intermediate variables!
    vo_eqns = [eqn for eqn in f_jaxpr.eqns if filter_literals(f_jaxpr.jaxpr._outvars) & filter_literals(eqn.invars)]
    o_eqns = [eqn for eqn in f_jaxpr.eqns if filter_literals(f_jaxpr.jaxpr._outvars) & filter_literals(eqn.outvars)]
    v_eqns = [eqn for eqn in f_jaxpr.eqns if eqn in vo_eqns or not eqn in o_eqns]
    return v_eqns, vo_eqns, o_eqns
    

def make_graph(f_jaxpr: Union[Jaxpr, Callable], 
               *xs: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    TODO this does not yet work as intended!!!
    """
    f_jaxpr = jax.make_jaxpr(f_jaxpr)(*xs) if isinstance(f_jaxpr, Callable) else f_jaxpr
    print(f_jaxpr)
    v_eqns, vo_eqns, o_eqns = filter_eqns(f_jaxpr)    
    
    
    # How to add a primitive to a jaxpr (not very useful here because it changes jaxpr)
    # prim = Primitive("add")
    # ins = [Literal(0., ShapedArray((), jnp.float32)), f_jaxpr.jaxpr._outvars[-1]]
    # outs = [Var(14, "", ShapedArray((), jnp.float32))]
    # e = JaxprEqn(ins, outs, prim, {}, set(), f_jaxpr.eqns[0].source_info)
    # f_jaxpr.eqns.append(e)
    # f_jaxpr.jaxpr._outvars = f_jaxpr.jaxpr._outvars[:-1] + outs
    # print(f_jaxpr)
    
    
    num_i = sum([aval.size for aval in f_jaxpr.in_avals])
    num_o = sum([aval.size for aval in f_jaxpr.out_avals])
    num_vo = sum([invar.aval.size for eqn in vo_eqns for invar in eqn.invars if invar in f_jaxpr.jaxpr._outvars])
    num_v = sum([outvar.aval.size for eqn in f_jaxpr.eqns for outvar in eqn.outvars if isinstance(outvar, Var)]) - num_o + num_vo
       
    info = make_graph_info([num_i, num_v, num_o])
    edges = make_empty_edges(info)
    
    # TODO reduce the complexity of this!
    intermediates = {}
    i, counter = 0, 0
    for eqn in v_eqns + o_eqns:
        for outvar in eqn.outvars:
            j = 0
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    # TODO optimize the if-conditions here!
                    if str(invar) not in intermediates:
                        intermediates[str(invar)] = counter
                        counter += invar.aval.size
                    elif eqn in vo_eqns:
                        for k in range(invar.aval.size):
                            j = intermediates[str(invar)]
                            edges = edges.at[j+k, i+k].set(1.)
                            continue
                    # make difference between vectorization and accumulation
                    if outvar.aval.size > 1 and invar.aval.size > 1:
                        # parallel op
                        for k in range(invar.aval.size):
                            j = intermediates[str(invar)]
                            edges = edges.at[j+k, i+k].set(1.) 
                    elif outvar.aval.size > 1 and invar.aval.size == 1:
                        # vectorized op
                        for k in range(outvar.aval.size):
                            j = intermediates[str(invar)]
                            edges = edges.at[j, i+k].set(1.)                      
                    else:
                        # accumulation op
                        for k in range(invar.aval.size):
                            j = intermediates[str(invar)]
                            edges = edges.at[j+k, i].set(1.)  
                    j += invar.aval.size
            i += outvar.aval.size
            
    return edges, info

