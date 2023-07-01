from typing import Callable, Tuple, Union, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import Var, ClosedJaxpr, JaxprEqn

import chex

from ..core import GraphInfo, make_empty_edges, make_graph_info
    
    
IGNORE = {lax.dynamic_slice_p, 
          lax.broadcast_in_dim_p,
            lax.dot_general_p, 
            lax.squeeze_p,
            # jax._src.pjit.pjit_p,
            lax.conv_general_dilated_p,
            lax.slice_p,
            lax.dynamic_update_slice_p}
    

def filter_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that filters out assignments of unused variables.
    """
    return [eqn for eqn in eqns if not str(eqn.outvars[0]) == "_"]


def make_graph(f_jaxpr: Union[ClosedJaxpr, Callable], *xs: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    Function that creates a computational graph from a JAX input function or a jaxpr.
    """
    jaxpr = jax.make_jaxpr(f_jaxpr)(*xs) if isinstance(f_jaxpr, Callable) else f_jaxpr
            
    num_i = sum([aval.size for aval in jaxpr.in_avals])
    num_o = sum([aval.size for aval in jaxpr.out_avals])
    eqns = filter_eqns(jaxpr.eqns)
    num_v = sum([outvar.aval.size for eqn in eqns for outvar in eqn.outvars])
       
    info = make_graph_info([num_i, num_v, num_o])
    edges = make_empty_edges(info)
    
    is_invar_list = []
    
    # Processing input variables
    variables = {}
    counter = 0
    for invar in jaxpr.jaxpr._invars:
        variables[str(invar)] = counter
        counter += invar.aval.size

    # Process intermediate variables
    i = 0
    for eqn in eqns:
        assert not eqn.primitive in IGNORE, "Primitive not supported by interpreter!"
        for outvar in eqn.outvars:
            if str(outvar) not in variables:
                variables[str(outvar)] = counter
                counter += outvar.aval.size
            j = 0
            for invar in eqn.invars:
                if invar in jaxpr.jaxpr._outvars:
                    is_invar_list.append(invar)
                if isinstance(invar, Var):
                    if outvar.aval.size > 1 and invar.aval.size > 1:
                        # Parallel op
                        for k in range(invar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j+k, i+k].set(1.) 
                    elif outvar.aval.size > 1 and invar.aval.size == 1:
                        # Vectorized op
                        for k in range(outvar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j, i+k].set(1.)                      
                    else:
                        # Accumulation op
                        for k in range(invar.aval.size):
                            j = variables[str(invar)]
                            edges = edges.at[j+k, i].set(1.)  
                    j += invar.aval.size
            i += outvar.aval.size
            
    # Processing output variables
    vertex_mask = jnp.zeros(num_v)
    k = 0
    for outvar in jaxpr.jaxpr._outvars:
        if not outvar in is_invar_list:
            idx = variables[str(outvar)]
            size = outvar.aval.size
            for i in range(size):
                vertex_mask = vertex_mask.at[k+i].set(idx+i-num_i+1)
            k += size
                
    # Make attention mask
    attn_mask = jnp.ones((num_v, num_v))
    return edges, info, vertex_mask, attn_mask

