from functools import wraps, partial
from collections import defaultdict

import time
import timeit

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax._src.util import safe_map
import jax._src.core as core

from jax.tree_util import tree_flatten, tree_unflatten
from graphax.interpreter.sparse_tensor import (SparseTensor, 
                                                DenseDimension, 
                                                SparseDimension)


def make_parallel_jacobian(primal, val_out, elemental):
    primal_size = len(primal.shape)
    out_size = len(val_out.shape)
        
    if primal_size == out_size:
        out_dims = [SparseDimension(i, val_out.aval.shape[i], i, out_size+i) 
                    for i, e in enumerate(val_out.aval.shape)]
        primal_dims = [SparseDimension(out_size+i, val_out.aval.shape[i], i, i) 
                    for i, e in enumerate(val_out.aval.shape)]
        # TODO add _eye_like here! - is a quick fix but does not tackle the core issue
        
    elif primal_size == 0:
        # Handling broadcast
        out_dims = [DenseDimension(i, val_out.aval.shape[i], i) 
                    for i, e in enumerate(val_out.aval.shape)]
        primal_dims = []
    else:
        out_dims = []
        primal_dims = []
    return SparseTensor(out_dims, primal_dims, elemental)


elemental_rules = {}


def defelemental(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental, elementalrule, primitive)


def standard_elemental(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(*primals, **params)
    elementals = elementals if type(elementals) is tuple else (elementals,)
    elementals_out = [make_parallel_jacobian(primal, val_out, elemental) 
                        for primal, elemental in zip(primals, elementals)]
    return val_out, elementals_out


# Useful for stuff such as exp_p
def defelemental2(primitive, elementalrule):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental2, elementalrule, primitive)


def standard_elemental2(elementalrule, primitive, primals, **params):
    assert elementalrule is not None
    val_out = primitive.bind(*primals, **params)
    elementals = elementalrule(val_out, *primals, **params)
    elementals = elementals if type(elementals) is tuple else (elementals,)
    elementals_out = [make_parallel_jacobian(primal, val_out, elemental) 
                        for primal, elemental in zip(primals, elementals)]
    return val_out, elementals_out
    
# Define elemental partials
defelemental(lax.neg_p, lambda x: -jnp.ones_like(x))
defelemental(lax.integer_pow_p, lambda x, y: y*x**(y-1))

defelemental2(lax.exp_p, lambda x: x)
defelemental(lax.log_p, lambda x: 1./x)

defelemental(lax.sin_p, lambda x: jnp.cos(x))
defelemental(lax.asin_p, lambda x: 1./jnp.sqrt(1.0 - x**2))
defelemental(lax.cos_p, lambda x: -jnp.sin(x))
defelemental(lax.acos_p, lambda x: -1./jnp.sqrt(1.0 - x**2))
defelemental(lax.tan_p, lambda x: 1.+jnp.tan(x)**2)
defelemental(lax.atan_p, lambda x: 1./(1. + x**2))

defelemental(lax.sinh_p, lambda x: jnp.cosh(x))
defelemental(lax.asinh_p, lambda x: jnp.sqrt(1. + x**2))
defelemental(lax.cosh_p, lambda x: jnp.sinh(x))
defelemental(lax.acosh_p, lambda x: 1./jnp.sqrt(x**2 - 1.))
defelemental(lax.tanh_p, lambda x: 1.-jnp.tanh(x)**2)
defelemental(lax.atanh_p, lambda x: 1./(1. - x**2))


def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))
    
defelemental(lax.add_p, add_elemental_rule)
    
    
def mul_elemental_rule(x, y):
    return (y, x)
    
defelemental(lax.mul_p, mul_elemental_rule)
   
        
def sub_elemental_rule(x, y):
    return (jnp.ones_like(y), -jnp.ones_like(x))
    
defelemental(lax.sub_p, sub_elemental_rule)
    
    
def div_elemental_rule(x, y):
    return (1./y, -x/y**2)

defelemental(lax.div_p, div_elemental_rule)


def transpose_elemental_rule(primals, **params):
    val_out = lax.transpose_p.bind(*primals, **params)
    
    permutation = params["permutation"]
    elemental = jnp.ones_like(val_out)
    
    new_out_dims, new_primal_dims = [], []
    
    for i, p in enumerate(permutation):
        new_out_dims.append(i, elemental.aval.shape[i], i, p)
        new_primal_dims.append(p, elemental.aval.shape[i], i, i)
    
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, elemental)]

elemental_rules[lax.transpose_p] = transpose_elemental_rule


def reduce_sum_elemental_rule(primals, **params):
    val_out = lax.reduce_sum_p.bind(*primals, **params)
    
    primal = primals[0]
    axes = params["axes"]
    if axes is None:
        axes = tuple(range(len(primal.shape)))
        new_out_dims.append(DenseDimension(0, 1, 0, True))
    elif type(axes) is int:
        axes = (axes,)
    new_out_dims, new_primal_dims = [], []
    _shape = []
    
    l = len(val_out.aval.shape)
    count = 0
    for i, size in enumerate(primal.aval.shape):
        if i in axes:
            new_primal_dims.append(DenseDimension(i, size, count))
            _shape.append(size)
            count += 1
        else:
            new_out_dims.append(SparseDimension(i, size, None, l+i))
            new_primal_dims.append(SparseDimension(l+i, size, None, i))
            
    val = jnp.ones(_shape)
    return val_out, [SparseTensor(new_out_dims, new_primal_dims, val)]
    
elemental_rules[lax.reduce_sum_p] = reduce_sum_elemental_rule


# TODO most important bit!
def dot_general_elemental_rule(primals, **params):
    val_out = lax.dot_general_p.bind(*primals, **params)
    lhs, rhs = primals
    
    # Which dimensions of the tensors are contracted
    dimension_numbers = params["dimension_numbers"][0]
    
    lhs_contracting_dims = dimension_numbers[0]
    rhs_contracting_dims = dimension_numbers[1]
    
    # TODO properly treat the transpose
    if len(rhs.aval.shape) > 1:
        transpose_rhs = rhs.T
        print(lhs, transpose_rhs)
        # TODO this needs generalization to higher-dimensional tensors
        transpose_rhs_dims = [1-d for d in dimension_numbers[1]]
    else:
        transpose_rhs = rhs
        transpose_rhs_dims = rhs_contracting_dims

    lhs_shape = list(lhs.aval.shape)
    rhs_shape = list(rhs.aval.shape)
    out_shape = list(val_out.aval.shape)
    
    lhs_jac_shape = out_shape + lhs_shape
    rhs_jac_shape = out_shape + rhs_shape
    
    lhs_primal_dims, rhs_primal_dims = [], []
    lhs_out_dims, rhs_out_dims = [], []
    
    i = 0
    for l, ld in enumerate(lhs_shape):
        _l = l + len(val_out.aval.shape)
        if l in lhs_contracting_dims:
            # Contracting dimension
            dim = transpose_rhs_dims[i] # TODO take account of the transpose
            lhs_primal_dims.append(DenseDimension(_l, transpose_rhs.aval.shape[dim], dim))
            i += 1
        else:
            lhs_primal_dims.append(SparseDimension(_l, lhs_jac_shape[i], None, l))
            lhs_out_dims.append(SparseDimension(l, lhs_jac_shape[i], None, _l))
            rhs_out_dims.append(DenseDimension(i, ld, l))
          
    j = 0   
    for r, rd in enumerate(rhs_shape):
        _r = r + len(val_out.aval.shape)
        if r in rhs_contracting_dims:
            # Contracting dimension
            dim = lhs_contracting_dims[j]
            rhs_primal_dims.append(DenseDimension(_r, lhs.aval.shape[dim], dim))
            j += 1
        else:
            rhs_primal_dims.append(SparseDimension(_r, rhs_jac_shape[j], None, r))
            rhs_out_dims.append(SparseDimension(r, rhs_jac_shape[j], None, _r))
            lhs_out_dims.append(DenseDimension(j, rd, transpose_rhs_dims.index(r)))
        
    lhs_tensor = SparseTensor(lhs_out_dims, lhs_primal_dims, transpose_rhs)
    rhs_tensor = SparseTensor(rhs_out_dims, rhs_primal_dims, lhs)   
    
    return val_out, [lhs_tensor, rhs_tensor]

elemental_rules[lax.dot_general_p] = dot_general_elemental_rule


def jacve(fun, order, argnums=(0,)):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        # Make repackaging work properly with one input value only
        flattened_args, in_tree = tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)
        out = vertex_elimination_jaxpr(closed_jaxpr.jaxpr, order, closed_jaxpr.literals, *args, argnums=argnums)
        out = tree_unflatten(in_tree, out)
        return out
    return wrapped


def vertex_elimination_jaxpr(jaxpr, order, consts, *args, argnums=(0,)):    
    env = {}
    graph = defaultdict(lambda: defaultdict())
    transpose_graph = defaultdict(lambda: defaultdict())

    # Reads variable and corresponding traced shaped array
    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    # Adds new variable and corresponding traced shaped array
    def write(var, val):
        env[var] = val
        
    def write_elemental(invar, outval):
        if isinstance(invar, core.Var):
            graph[invar][eqn.outvars[0]] = outval
            transpose_graph[eqn.outvars[0]][invar] = outval
                
    jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]
    ignore_invars = [invar for i, invar in enumerate(jaxpr.invars) if i not in argnums]
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop though elemental partials
    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if eqn.primitive not in elemental_rules:
            raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")

        primal_outvals, elemental_outvals = elemental_rules[eqn.primitive](invals, **eqn.params)
        safe_map(write, eqn.outvars, [primal_outvals])

        safe_map(write_elemental, eqn.invars, elemental_outvals)
        
    if type(order) is str:
        if order == "forward" or order == "fwd":
            order = [i for i, eqn in enumerate(jaxpr.eqns) 
                    if eqn.outvars[0] not in jaxpr.outvars]
        elif order == "reverse" or order == "rev":
            # TODO reverse the order here!
            order = [i for i, eqn in enumerate(jaxpr.eqns) 
                    if eqn.outvars[0] not in jaxpr.outvars]
        else:
            raise ValueError(order + " is not a valid order identifier!")

    # Eliminate the vertices
    for vertex in order:
        # print(vertex)
        eqn = jaxpr.eqns[vertex-1]
        for out_edge in graph[eqn.outvars[0]].keys():
            out_val = graph[eqn.outvars[0]][out_edge]
            for in_edge in transpose_graph[eqn.outvars[0]].keys():
                in_val = transpose_graph[eqn.outvars[0]][in_edge]
                # print(in_edge, eqn.outvars[0], out_edge)
                edge_outval = out_val * in_val
                # print("mul shape", edge_outval)

                if graph.get(in_edge).get(out_edge) is not None:
                    _edge = transpose_graph[out_edge][in_edge]
                    # print("add shape", _edge)
                    edge_outval += _edge
                graph[in_edge][out_edge] = edge_outval
                transpose_graph[out_edge][in_edge] = edge_outval
                
        for in_edge in transpose_graph[eqn.outvars[0]].keys():
            del graph[in_edge][eqn.outvars[0]]
        for out_edge in graph[eqn.outvars[0]].keys():    
            del transpose_graph[out_edge][eqn.outvars[0]]
        
        # Cleanup eliminated vertices            
        del graph[eqn.outvars[0]]
        del transpose_graph[eqn.outvars[0]]    

    # Collect outputs TODO replace excluded argnums with Nones in pytree
    jac_vals = [graph[invar][outvar].materialize_actual_shape() 
                for outvar in jaxpr.outvars for invar in jaxpr.invars]
    
    # Restructure Jacobians for more complicated pytrees
    n = len(jaxpr.outvars)
    if n > 1:
        ratio = len(jac_vals)//len(jaxpr.outvars)
        jac_vals = [tuple(jac_vals[i*n:i*n+n]) for i in range(0, ratio)]

    return jac_vals

