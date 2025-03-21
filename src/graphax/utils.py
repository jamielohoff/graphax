from typing import Sequence, Set

import jax
import jax.core as core
import jax.numpy as jnp


def get_output_vertices(jaxpr: core.Jaxpr) -> Set[int]:
    """
    Function that returns a set containing all output vertices of the given
    computational graph/jaxpr.
    """
    vo_vertices = set()
    var_id = {}
    
    for eqn in jaxpr.eqns:
        for outvar in eqn.outvars:
            if type(outvar) is core.Var and outvar not in var_id.keys():
                var_id[outvar] = counter
                counter += 1
        
        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)
    return vo_vertices


def get_valid_vertices(jaxpr: core.Jaxpr) -> Sequence[int]:
    """
    Function that checks if the supplied elimination order is valid for the 
    given computational graph/jaxpr. In the case of an elimination order that
    """
    vo_vertices = get_output_vertices(jaxpr)
    return [i for i, eqn in enumerate(jaxpr.eqns, start=1) 
            if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices]

