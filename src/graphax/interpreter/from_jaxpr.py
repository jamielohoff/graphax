from typing import Callable, Tuple, Union, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import Var, ClosedJaxpr, JaxprEqn

import chex

from ..core import GraphInfo, make_empty_edges, make_graph_info
    

def filter_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that filters out assignments of unused variables.
    """
    return [eqn for eqn in eqns if not str(eqn.outvars[0]) == "_"]


def populate_attn_mask(mask: chex.Array, output_vertices: chex.Array):
    def loop_fn(carry, idx):
        _mask = carry
        _mask = _mask.at[idx-1, :].set(0.)
        _mask = _mask.at[:, idx-1].set(0.)
        return _mask, None
    
    output, _ = lax.scan(loop_fn, mask, output_vertices)
    return output


def make_graph(f_jaxpr: Union[ClosedJaxpr, Callable], 
               *xs: chex.Array) -> Tuple[chex.Array, GraphInfo]:
    """
    Function that creates a computational graph from a pure JAX input function
    or a jaxpr.
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
    output_vertices = jnp.zeros(num_v)
    k = 0
    for outvar in jaxpr.jaxpr._outvars:
        if not outvar in is_invar_list:
            idx = variables[str(outvar)]
            size = outvar.aval.size
            for i in range(size):
                output_vertices = output_vertices.at[k+i].set(idx-size+i+1)
                
    # Make attention mask
    attn_mask = jnp.ones((num_v, num_v))
    # attn_mask = populate_attn_mask(attn_mask, output_vertices.astype(jnp.int32))
    return edges, info, output_vertices, attn_mask

