from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.lax as lax

import numpy as np
import scipy as scp
from scipy.sparse import lil_matrix

id_p = jax.core.Primitive("id")
def id_prim(x, y):
    return id_p.bind(x, y)

def id_impl():
    pass


def f(x, y):
    z = x + y
    w = jnp.sin(z)
    return z*w, 3+w

f_jaxpr = jax.make_jaxpr(f)(1., 2.)

elemental_registry = {}

# TODO define additional primitives
id_p = lax.standard_naryop([{np.floating}], "id")
# unops are easy to handle
elemental_registry[lax.exp_p] = lax.exp_p
elemental_registry[lax.sin_p] = lax.cos_p

elemental_registry[lax.mul_p] = id_p # Handle this case
elemental_registry[lax.add_p] = lax.add_p # Handle this case


class JaxprGraph:
    jaxpr: jax.core.Jaxpr
    graph_jaxpr: jax.core.Jaxpr
    
    vertices: Sequence[int]
    invertices: Sequence[int] 
    outvertices: Sequence[int]   
    edges: Sequence[Tuple[int]] 
    idx: int
    
    def __init__(self, jaxpr) -> None:
        self.jaxpr = jaxpr
        
        self.graph_jaxpr = jax.core.Jaxpr(
            constvars=[],
            invars=self.jaxpr.invars,
            outvars=[],
            eqns=self.jaxpr.eqns
        )


        self.invertices = [v.count for v in self.jaxpr.invars]
        self.outvertices = [v.count for v in self.jaxpr.outvars]
        
        n = len(self.invertices) + len(self.jaxpr.eqns)
        self.edges = lil_matrix((n, n), dtype=np.int32)
        
        self.idx = n
        for eqn in self.jaxpr.eqns:
            # TODO this double loop is non-sense! 
            for atom in eqn.invars:
                for var in eqn.outvars:
                    # Only variables should be part of the computational graph
                    if isinstance(atom, jax.core.Var): 
                        prim = elemental_registry[eqn.primitive]
                        
                        self.edges[atom.count, var.count] = self.idx # we can save this as a sparse matrix!
                        
                        invars = eqn.invars
                        outvars = [jax.core.Var(self.idx, str((atom.count, var.count)), var.aval)]
                        
                        jaxpr_eqn = jax.core.JaxprEqn(invars, outvars, prim, {}, None, None)
                        self.graph_jaxpr.eqns.append(jaxpr_eqn)
                        self.idx += 1
                    
    def front_eliminate(self, edge: Tuple[int]):
        """TODO add docstring

        Args:
            edge (Tuple[int]): _description_
        """
        ev = self.edges[edge[0], edge[1]]
        edge_vars = self.graph_jaxpr.eqns[ev-len(self.invertices)]
        self.edges[edge[0], edge[1]] = 0.
        
        for i, j in zip(*self.edges.nonzero()):
            v = self.edges[i, j]
            vars = self.graph_jaxpr.eqns[v-len(self.invertices)]
            v_aval = vars.outvars[0].aval
            if i == edge[1]:              
                mul_prim = lax.mul_p
                invars = edge_vars.outvars + vars.outvars
                outvars = [jax.core.Var(self.idx, "", v_aval)]
                mul_eqn = jax.core.JaxprEqn(invars, outvars, mul_prim, {}, None, None)
                self.graph_jaxpr.eqns.append(mul_eqn)
                self.idx += 1
                
                add_prim = lax.add_p
                # Check whether edge already exists
                _v = self.edges[edge[0], j]
                if _v != 0:
                    _vars = self.graph_jaxpr.eqns[_v-len(self.invertices)] 
                    invars = _vars.outvars + outvars
                else:
                    invars = [jax.core.Literal(0., v_aval)] + outvars
                    # create new edge if it does not exist yet
                
                outvars = [jax.core.Var(self.idx, "", v_aval)]
                add_eqn = jax.core.JaxprEqn(invars, outvars, add_prim, {}, None, None)
                self.graph_jaxpr.eqns.append(add_eqn)
                self.edges[edge[0], j] = 0. # this has to be explicit!
                self.edges[edge[0], j] = self.idx
                self.idx += 1
                
    def back_eliminate(self, edge: Tuple[int]):
        """TODO add docstring

        Args:
            edge (Tuple[int]): _description_
        """
        ev = self.edges[edge[0], edge[1]]
        edge_vars = self.graph_jaxpr.eqns[ev-len(self.invertices)]
        self.edges[edge[0], edge[1]] = 0.
        
        for i, j in zip(*self.edges.nonzero()):
            v = self.edges[i, j]
            vars = self.graph_jaxpr.eqns[v-len(self.invertices)]
            v_aval = vars.outvars[0].aval
            if j == edge[0]:                
                mul_prim = lax.mul_p
                invars = edge_vars.outvars + vars.outvars
                outvars = [jax.core.Var(self.idx, "", v_aval)]
                mul_eqn = jax.core.JaxprEqn(invars, outvars, mul_prim, {}, None, None)
                self.graph_jaxpr.eqns.append(mul_eqn)
                self.idx += 1
                
                add_prim = lax.add_p
                # Check whether edge already exists
                _v = self.edges[i, edge[1]]
                if _v != 0:
                    _vars = self.graph_jaxpr.eqns[_v-len(self.invertices)] 
                    invars = _vars.outvars + outvars
                else:
                    invars = [jax.core.Literal(0., v_aval)] + outvars
                    # create new edge if it does not exist yet
                
                outvars = [jax.core.Var(self.idx, "", v_aval)]
                add_eqn = jax.core.JaxprEqn(invars, outvars, add_prim, {}, None, None)
                self.graph_jaxpr.eqns.append(add_eqn)
                self.edges[i, edge[1]] = 0. # this has to be explicit!
                self.edges[i, edge[1]] = self.idx
                self.idx += 1
                
    def eliminate(self, vertex: int):
        """TODO add docstring

        Args:
            vertex (int): _description_
        """
        for i, j in zip(*self.edges.nonzero()):
            # front-eliminiation of ingoing edges
            if j == vertex:
                self.front_eliminate((i,j))
                
            # back-elimination of outgoing edges
            if i == vertex:
                self.back_eliminate((i,j))
                    
    def finalize(self):
        """TODO add docstring

        Args:
            edge (Tuple[int]): _description_
        """
        for i in self.invertices:
            for j in self.outvertices:
                v = self.edges[i,j]
                outvars = self.graph_jaxpr.eqns[v-len(self.invertices)].outvars
                self.graph_jaxpr.outvars.append(*outvars)
        self.graph_jaxpr = jax.core.ClosedJaxpr(self.graph_jaxpr, [])        
        
    def to_function(self):
        pass
        

jg = JaxprGraph(f_jaxpr.jaxpr)  
jg.front_eliminate((2,3))
# jg.front_eliminate((0,2))
# jg.front_eliminate((1,2))
# jg.back_eliminate((2,3))
# jg.back_eliminate((2,4))
# jg.eliminate(3)
# jg.eliminate(2)
jg.finalize()
print(jg.graph_jaxpr)
print(jg.edges)