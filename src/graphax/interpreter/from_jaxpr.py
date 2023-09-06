from typing import Callable, Sequence, Union

import jax
from jax._src.core import ClosedJaxpr, JaxprEqn, Literal

from chex import Array

from ..core import make_empty_edges, get_shape
from .utils import add_slice
from .prim_mapper import vertex_registry
    

def filter_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that filters out assignments of unused variables.
    """
    return [eqn for eqn in eqns if not str(eqn.outvars[0]) == "_"]


def count_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that unrolls "pjit" to count the number of equations in a jaxpr.
    """
    filtered_eqns = []
    for eqn in eqns:
        if not str(eqn.outvars[0]) == "_":
            if eqn.primitive.name == "pjit":
                subeqns = unroll_pjit(eqn)
                filtered_eqns.extend(subeqns)
            else:
                filtered_eqns.append(eqn)
                
    return len(filtered_eqns)


def unroll_pjit(eqn: JaxprEqn) -> Sequence[JaxprEqn]:
    """
    Function that unrolls "pjit" primitives which define subexpressions that
    are not tracked by the algorithm otherwise
    """
    eqns = []
    subjaxpr = eqn.params["jaxpr"]
    for subeqn in subjaxpr.eqns:
        if subeqn.primitive.name == "pjit":
            subeqns = unroll_pjit(subeqn)
            eqns.extend(subeqns)
        else:
            eqns.append(subeqn)
    return eqns


def make_graph(f_jaxpr: Union[ClosedJaxpr, Callable], *xs: Array) -> Array:
    """
    Function that creates a computational graph from a JAX input function or a jaxpr.
    """
    jaxpr = jax.make_jaxpr(f_jaxpr)(*xs) if isinstance(f_jaxpr, Callable) else f_jaxpr
            
    num_i = len(jaxpr.jaxpr._invars)
    num_o = len(jaxpr.jaxpr._outvars)
    eqns = filter_eqns(jaxpr.eqns)
    num_v = len(eqns)
       
    edges = make_empty_edges([num_i, num_v, num_o])
    edges = edges.at[0, 0, 0].set(num_i)
    edges = edges.at[0, 0, 1].set(num_v-num_o)
    edges = edges.at[0, 0, 2].set(num_o)
        
    is_invar_list = []
    
    # Processing input variables    
    variables = {}
    counter = 1
    for invar in jaxpr.jaxpr._invars:
        variables[str(invar)] = counter
        counter += 1

    # Process intermediate variables
    for eqn in eqns:
        is_invar_list.extend(eqn.invars)
        # Ignore calculation with just literals, i.e. constant values
        for outvar in eqn.outvars:
            # Add new variables to tracker
            if str(outvar) not in variables:
                variables[str(outvar)] = counter
                counter += 1
                        
        # Resolves primitive and adds it to the edge representation matrix
        add_vertex_fn = vertex_registry[eqn.primitive]
        edges = add_vertex_fn(edges, eqn, variables)      
     
    # Processing output variables
    for outvar in jaxpr.jaxpr._outvars:
        if type(outvar) is Literal: continue
        num_i, num_vo = get_shape(edges)
        idx = variables[str(outvar)]
        if outvar in is_invar_list:
            # Track which vertices are output vertices but also 
            # intermediate vertices. These are eliminated non-the-less, 
            # but we add an additional slice to the tensor with a copy-gradient 
            # edge
            edges = add_slice(edges, outvar, idx, num_i, num_vo)
        else:
            edges = edges.at[1, 0, idx-num_i-1].set(1)
            edges = edges.at[2, 0, idx-num_i-1].set(1)

    return edges

