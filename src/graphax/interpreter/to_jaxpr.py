from typing import Dict, Sequence, Tuple
from collections import defaultdict

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax.core import (ClosedJaxpr, 
                    Jaxpr, 
                    JaxprEqn, 
                    Primitive, 
                    eval_jaxpr, 
                    new_jaxpr_eqn, 
                    Var, 
                    Atom, 
                    Literal, 
                    ShapedArray)


class ElementalPartial:
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        pass
    
class ElementalExp(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        return eqn.outvars, None
    
class ElementalSin(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        outvars = [Var(count, "_d", ShapedArray((), jnp.float32))]
        return outvars, new_jaxpr_eqn(eqn.invars, outvars, lax.cos_p, {}, set(), source_info=None)
    
class ElementalAdd(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        # TODO improve this for vectorized edges
        return [Literal(1., ShapedArray((), jnp.float32))], None
    
class ElementalMul(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        # swap invars and outvars here
        outvars = list(set(eqn.invars) - set([invar]))
        return outvars, None
    
class ElementalTan(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        # swap invars and outvars here
        outvars = list(set(eqn.invars) - set([invar]))
        return outvars, None
    
class ElementalDiv(ElementalPartial):
    def get_jaxpr(self, invar, eqn, count) -> Sequence[JaxprEqn]:
        # swap invars and outvars here
        outvars = list(set(eqn.invars) - set([invar]))
        return outvars, None
    
elemental_registry = {}

elemental_registry[lax.exp_p] = ElementalExp()
elemental_registry[lax.sin_p] = ElementalSin()
elemental_registry[lax.add_p] = ElementalAdd()
elemental_registry[lax.mul_p] = ElementalMul()

# def f(x, y):
#     z = x * y
#     w = jnp.sin(z)
#     return w + z, jnp.exp(z)

def f(nu, gamma, omega, t):
        y1 = nu*jnp.tan(omega*t)/(gamma-jnp.tan(omega*t))
        y2 = gamma*y1
        return y1, y2

jaxpr = jax.make_jaxpr(f)(2., 2., 2., 2.)
jac_f = jax.jacrev(f, argnums=(0, 1, 2, 3))
print(jac_f(2., 2., 2., 2.))
jac_jaxpr = jax.make_jaxpr(jac_f)(2., 2., 2., 2.)
print(jac_jaxpr)

class ComputationalGraph:
    edges: Dict
    jaxpr: ClosedJaxpr
    count: int
    num_invars: int
    derivative_code: Sequence[JaxprEqn]
    
    def __init__(self, jaxpr: ClosedJaxpr) -> None:
        self.jaxpr = jaxpr
        self.count = len([eqn.outvars[0] for eqn in jaxpr.jaxpr._eqns]) + len(jaxpr.jaxpr._invars)
        self.num_invars = len(self.jaxpr.jaxpr._invars)
        self.in_edges = {i: {} for i in range(-self.num_invars+1, len(jaxpr.eqns)+1)}
        self.out_edges = {i: {} for i in range(-self.num_invars+1, len(jaxpr.eqns)+1)}
        self.derivative_code = []
        
        for j, eqn in enumerate(self.jaxpr.eqns, 1):
            # Add "global" incoming edges to the vertex if the exist
            for i, invar in enumerate(self.jaxpr.jaxpr._invars, -self.num_invars+1):
                if invar in eqn.invars:
                    if j in self.out_edges[i]:
                        self.in_edges[j][i] = self.out_edges[i][j]
                    else: 
                        derivative = elemental_registry[eqn.primitive]
                        partials, jaxpreqn = derivative.get_jaxpr(invar, eqn, self.count)
                        self.in_edges[j][i] = partials
                        self.out_edges[i][j] = partials
                        if jaxpreqn is not None :
                            self.derivative_code.append(jaxpreqn)
                            self.count += 1
            
            # Add incoming edges to the vertex
            for i, _eqn in enumerate(self.jaxpr.eqns[:j], 1):
                if len(set(eqn.invars) & set(_eqn.outvars)) != 0:
                    if j in self.out_edges[i]:
                        self.in_edges[j][i] = self.out_edges[i][j]
                    else: 
                        derivative = elemental_registry[eqn.primitive]
                        partials, jaxpreqn = derivative.get_jaxpr(_eqn.outvars[0], eqn, self.count)
                        self.in_edges[j][i] = partials
                        self.out_edges[i][j] = partials
                        if jaxpreqn is not None :
                            self.derivative_code.append(jaxpreqn)
                            self.count += 1
                       
            # Add outgoing edges to the vertex
            for k, _eqn in enumerate(self.jaxpr.eqns[j+1:], j+1):
                if len(set(eqn.invars) & set(_eqn.outvars)) != 0:
                    if j in self.in_edges[k]:
                        self.out_edges[j][k] = self.in_edges[k][j] 
                    else:
                        derivative = elemental_registry[_eqn.primitive]
                        partials, jaxpreqn = derivative.get_jaxpr(eqn.outvars[0], _eqn, self.count)
                        self.in_edges[k][j] = partials
                        self.out_edges[j][k] = partials
                        if jaxpreqn is not None:
                            self.derivative_code.append(jaxpreqn)
                            self.count += 1 
    
    def build(self):
        self.jaxpr.jaxpr._outvars = []
        # Package the components of the Jacobian
        for i in range(-self.num_invars+1, 1):
            invars = []
            for j in self.out_edges[i].keys():
                invar = self.out_edges[i][j]
                if invar[0].aval.shape == ():
                    _invars = [Var(self.count, "_d", ShapedArray((1,), jnp.float32))]
                    eqn = new_jaxpr_eqn(invar, _invars, lax.broadcast_in_dim_p, {"broadcast_dimensions":(), "shape":(1,)}, set(), source_info=None)
                    invars.extend(_invars)
                    self.derivative_code.append(eqn)
                    self.count += 1
                else: 
                    invars.extend(invar)
            l = len(self.out_edges[i].keys())
            outvars = [Var(self.count, "_d", ShapedArray((l,), jnp.float32))]
            eqn = new_jaxpr_eqn(invars, outvars, lax.concatenate_p, {"dimension": 0}, set(), source_info=None)
            self.derivative_code.append(eqn)
            self.jaxpr.jaxpr._outvars.extend(outvars)
            self.count += 1

        # Add the derivative code
        self.jaxpr.jaxpr._eqns.extend(self.derivative_code)
        return jaxpr 
    
    def vertex_eliminate(self, j: int):
        del_in_idxs = set()
        del_out_idxs = set()
        for i in self.in_edges[j].keys():
            for k in self.out_edges[j].keys():
                # Multiply the pre-partial and post-partial results
                outvars = [Var(self.count, "_d", ShapedArray((), jnp.float32))]
                invars = self.in_edges[j][i] + self.out_edges[j][k]
                multiplication = new_jaxpr_eqn(invars, outvars, lax.mul_p, {}, set(), source_info=None)
                self.derivative_code.append(multiplication)
                self.count += 1
                
                # Check if we have an additional addition because there already exists a direct edge
                if i in self.in_edges[k].keys():
                    invars = self.in_edges[k][i] + outvars
                    outvars = [Var(self.count, "_d", ShapedArray((), jnp.float32))]
                    addition = new_jaxpr_eqn(invars, outvars, lax.add_p, {}, set(), source_info=None)
                    self.derivative_code.append(addition)
                    self.count += 1
                    
                self.out_edges[i][k] = outvars
                self.in_edges[k][i] = outvars
                del_in_idxs.add((k, j))
                del_out_idxs.add((i, j))
        # Delete all the edges of the respective vertex
        for iidx in del_in_idxs:
            del self.in_edges[iidx[0]][iidx[1]]
        for oidx in del_out_idxs:
            del self.out_edges[oidx[0]][oidx[1]]
        del self.in_edges[j]
        del self.out_edges[j]
        
    def front_eliminate(self):
        pass
    
    def back_eliminate(self):
        pass


graph = ComputationalGraph(jaxpr)
for i in range(1, 9):
    graph.vertex_eliminate(i)
print(graph.out_edges)
new_jaxpr = graph.build()
print(new_jaxpr)

import time

f = lambda x, y: eval_jaxpr(new_jaxpr.jaxpr, new_jaxpr.literals, x, y)
f_jit = jax.jit(f)
print(f_jit(2., 2., 2., 2.))
st = time.time()
for _ in range(100000):
    f_jit(2., 2., 2., 2.)
print(time.time() - st)

