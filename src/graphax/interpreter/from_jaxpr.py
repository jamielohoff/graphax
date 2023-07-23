from typing import Callable, Union, Sequence

import jax
from jax._src.core import ClosedJaxpr, JaxprEqn

from chex import Array

from graphax.core import make_empty_edges
from graphax.interpreter.prim_mapper import vertex_registry, filter_invars
    

def filter_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that filters out assignments of unused variables.
    """
    return [eqn for eqn in eqns if not str(eqn.outvars[0]) == "_"]


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
    edges = edges.at[0, 0, 1].set(num_v)
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
        filtered_invars = filter_invars(eqn, variables)
        # Ignore calculation with just Literals, i.e. constant values
        for outvar in eqn.outvars:
            # Add new variables to tracker
            if str(outvar) not in variables:
                if len(filtered_invars) == 0:
                    # If a variable is the result of computations with only 
                    # Literals, we also treat it as a literal
                    variables[str(outvar)] = -1
                else:
                    variables[str(outvar)] = counter
                    counter += 1
                        
        # Resolves primitive and adds it to the edge representation matrix
        try:
            add_vertex_fn = vertex_registry[eqn.primitive]
            edges = add_vertex_fn(edges, eqn, variables)
        except:
            ValueError("Primitive", eqn.primitive, "not supported!")         
     
    # Processing output variables
    for outvar in jaxpr.jaxpr._outvars:
        if not outvar in is_invar_list:
            # Track which vertices are output vertices
            idx = variables[str(outvar)]
            edges = edges.at[1, 0, idx-num_i-1].set(1)
            edges = edges.at[2, 0, idx-num_i-1].set(1)

    return edges

