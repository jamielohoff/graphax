from functools import wraps, partial, reduce
from collections import defaultdict

import timeit

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax._src.util import safe_map
from jax._src.ad_util import Zero
import jax._src.core as core

from jax.tree_util import tree_flatten, tree_unflatten


elemental_rules = {}


def defelemental(primitive, *elementalrules):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental, elementalrules, primitive)


def standard_elemental(elementalrules, primitive, primals, **params):
    val_out = primitive.bind(*primals, **params)
    
    elementals_out = [rule(*primals, **params) for rule in elementalrules 
                    if rule is not None]
    return val_out, reduce(add_elementals, elementals_out, Zero.from_value(val_out))


# Useful for stuff such as exp_p
def defelemental2(primitive, *elementalrules):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_rules[primitive] = partial(standard_elemental2, elementalrules, primitive)


def standard_elemental2(partialrules, primitive, primals, **params):
    val_out = primitive.bind(*primals, **params)
    
    elementals_out = [rule(val_out, *primals, **params) for rule in partialrules 
                    if rule is not None]
    return val_out, reduce(add_elementals, elementals_out, Zero.from_value(val_out))


def add_elementals(x, y):
    if type(x) is Zero:
        return y
    elif type(y) is Zero:
        return x
    else:
        return jax._src.ad_util.add_jaxvals(x, y)
   

def _eye_like1(val):
    if hasattr(val, "aval"):
        if type(val.aval) is core.ShapedArray:
            size = val.size
            if size > 1:
                return jnp.eye(size, size).reshape(*val.shape, *val.shape)
    return 1.


def _eye_like(in_val, out_val):
    if hasattr(in_val, "aval") and hasattr(out_val, "aval"):
        if type(in_val.aval) is core.ShapedArray and type(out_val.aval) is core.ShapedArray:
            in_size = in_val.size
            out_size = out_val.size
            if in_size > 1 and out_size > 1:
                return jnp.eye(out_size, in_size).reshape(*out_val.shape, *in_val.shape)
            elif in_size > 1 and out_size == 1:
                return jnp.ones((1, in_size))
            elif in_size == 1 and out_size > 1:
                return jnp.ones(out_size)
    return 1.


# Manages the multiplication of Jacobians
# TODO This guy needs an ungodly amount of optimization
def _mul(lhs, rhs, vertex):
    if vertex.aval.ndim > 0 and hasattr(lhs, "aval") and hasattr(rhs, "aval"):
        if vertex.aval.ndim == 1:
            if lhs.aval.ndim == 1:
                if rhs.aval.ndim == 1:
                    return jnp.einsum("k,k->", lhs, rhs)
                elif rhs.aval.ndim == 2:
                    return jnp.einsum("k,km->m", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("k,kmn->mn", lhs, rhs)
            elif lhs.aval.ndim == 2:
                if rhs.aval.ndim == 1:
                    return jnp.einsum("ik,k->i", lhs, rhs)
                elif rhs.aval.ndim == 2:
                    return jnp.einsum("ik,km->im", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("ik,kmn->imn", lhs, rhs)
            elif lhs.aval.ndim == 3:
                if rhs.aval.ndim == 1:
                    return jnp.einsum("ijk,k->ij", lhs, rhs)
                elif rhs.aval.ndim == 2:
                    return jnp.einsum("ijk,km->ijm", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("ijk,kmn->ijmn", lhs, rhs)
            
        elif vertex.aval.ndim == 2:
            if lhs.aval.ndim == 2:
                if rhs.aval.ndim == 2:
                    return jnp.einsum("kl,kl->", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("kl,klm->m", lhs, rhs)
                elif rhs.aval.ndim == 4:
                    return jnp.einsum("kl,klmn->mn", lhs, rhs)
            elif lhs.aval.ndim == 3:
                if rhs.aval.ndim == 2:
                    return jnp.einsum("ikl,kl->i", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("ikl,klm->im", lhs, rhs)
                elif rhs.aval.ndim == 4:
                    return jnp.einsum("ikl,klmn->imn", lhs, rhs)
            elif lhs.aval.ndim == 4:
                if rhs.aval.ndim == 2:
                    return jnp.einsum("ijkl,kl->ij", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("ijkl,klm->ijm", lhs, rhs)
                elif rhs.aval.ndim == 4:
                    return jnp.einsum("ijkl,klmn->ijmn", lhs, rhs)
    elif hasattr(lhs, "aval") and hasattr(rhs, "aval"):
        if lhs.aval.ndim == 1:
            if rhs.aval.ndim == 1:
                return jnp.einsum("i,j->ij", lhs, rhs)
            if rhs.aval.ndim == 2:
                return (lhs*rhs).T
        elif lhs.aval.ndim == 2:
            if rhs.aval.ndim == 1:
                return (lhs*rhs).T
            if rhs.aval.ndim == 2:
                return jnp.einsum("ij,kl->ijkl", lhs, rhs)
    return lhs*rhs
    
# Define elemental partials
defelemental(lax.neg_p, lambda x: -_eye_like1(x))
defelemental(lax.integer_pow_p, lambda x, y: _eye_like1(x)*y*x**(y-1))

defelemental2(lax.exp_p, lambda x: _eye_like1(x)*x)
defelemental(lax.log_p, lambda x: _eye_like1(x)/x)

defelemental(lax.sin_p, lambda x: _eye_like1(x)*jnp.cos(x))
defelemental(lax.asin_p, lambda x: _eye_like1(x)/jnp.sqrt(1.0 - x**2))
defelemental(lax.cos_p, lambda x: -_eye_like1(x)*jnp.sin(x))
defelemental(lax.acos_p, lambda x: -_eye_like1(x)/jnp.sqrt(1.0 - x**2))
defelemental(lax.tan_p, lambda x: _eye_like1(x) + _eye_like1(x)*jnp.tan(x)**2)
defelemental(lax.atan_p, lambda x: _eye_like1(x)/(1.0 + x**2))

defelemental(lax.sinh_p, lambda x: _eye_like1(x)*jnp.cosh(x))
defelemental(lax.asinh_p, lambda x: _eye_like1(x)/jnp.sqrt(1.0 + x**2))
defelemental(lax.cosh_p, lambda x: _eye_like1(x)*jnp.sinh(x))
defelemental(lax.acosh_p, lambda x: _eye_like1(x)/jnp.sqrt(x**2 - 1.0))
defelemental(lax.tanh_p, lambda x: _eye_like1(x) - _eye_like1(x)*jnp.tanh(x)**2)
defelemental(lax.atanh_p, lambda x: _eye_like1(x)/(1.0 - x**2))

def add_elemental_rule(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (eye, eye)
    else:
        return (_eye_like(y, out), _eye_like(x, out))
    
defelemental2(lax.add_p, add_elemental_rule)
    
    
def mul_elemental_rule(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (y*eye, x*eye)
    else:
        return (y*_eye_like(x, out), x*_eye_like(y, out))
    
defelemental2(lax.mul_p, mul_elemental_rule)
   
        
def sub_elemental_rule(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (eye, eye)
    else:
        return (_eye_like(y, out), -_eye_like(x, out))
    
defelemental2(lax.sub_p, sub_elemental_rule)
    
    
def div_elemental_rule(out, x, y, dimension_numbers):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (eye/y, -eye*x/y**2)
    else:
        return (_eye_like(x, out)/y, -_eye_like(y, out)*x/y**2)

defelemental2(lax.div_p, div_elemental_rule)

# TODO This guy needs some optimization
def transpose_elemental_rule(inval, permutation):
    if inval.ndim == 2:
        shape0 = inval.shape[0]
        shape1 = inval.shape[1]
        eye0 = jnp.eye(shape0, shape0)
        eye1 = jnp.eye(shape1, shape1)
        return jnp.einsum("ik,jl->ijlk", eye1, eye0)
    return jnp.ones((1, inval.ndim))

defelemental(lax.transpose_p, transpose_elemental_rule) 


defelemental2(lax.reduce_sum_p, lambda out, x, axes: jnp.outer(_eye_like1(out), jnp.ones_like(x)))

  
# TODO most important bit!
def dot_general_elemental_rule(out, lhs, rhs, 
                                dimension_numbers, 
                                precision, 
                                preferred_element_type):
    if out.ndim == 1:
        lhs_eye = jnp.eye(out.shape[0], out.shape[0])
        jac_lhs = jnp.einsum("ik,j->kij", lhs_eye, rhs)
        
        return jac_lhs, lhs
    else:
        lhs_eye = jnp.eye(out.shape[0], out.shape[0])
        jac_lhs = jnp.einsum("ik,jl->klij", lhs_eye, rhs)
        
        rhs_eye = jnp.eye(out.shape[1], out.shape[1])
        jac_rhs = jnp.einsum("ik,jl->ilkj", lhs, rhs_eye, )
        return jac_lhs, jac_rhs

defelemental2(lax.dot_general_p, dot_general_elemental_rule)


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
                
    # jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]
    # ignore_invars = [invar for i, invar in enumerate(jaxpr.invars) if i not in argnums]
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # Loop though elemental partials
    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if eqn.primitive not in elemental_rules:
            raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")
        
        primal_outvals, elemental_outvals = elemental_rules[eqn.primitive](invals, **eqn.params)
        safe_map(write, eqn.outvars, [primal_outvals])

        elemental_outvals = elemental_outvals if type(elemental_outvals) is tuple else [elemental_outvals]
        safe_map(write_elemental, eqn.invars, elemental_outvals)

    # Eliminate the vertices
    for vertex in order:
        eqn = jaxpr.eqns[vertex-1]
        for out_edge in graph[eqn.outvars[0]].keys():
            out_val = graph[eqn.outvars[0]][out_edge]
            for in_edge in transpose_graph[eqn.outvars[0]].keys():
                in_val = transpose_graph[eqn.outvars[0]][in_edge]
                # print(eqn.outvars[0], out_edge, out_val.shape)
                # print(in_edge, eqn.outvars[0], in_val.shape)
                edge_outval = _mul(out_val, in_val, eqn.outvars[0])

                if graph.get(in_edge).get(out_edge) is not None:
                    _edge = transpose_graph[out_edge][in_edge]
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

    # Collect outputs
    # TODO this can be optimized as well
    jac_vals = [graph[invar][outvar] for outvar in jaxpr.outvars for invar in jaxpr.invars]
    
    # Restructure Jacobians for more complicated pytrees
    n = len(jaxpr.outvars)
    if n > 1:
        ratio = len(jac_vals)//len(jaxpr.outvars)
        jac_vals = [tuple(jac_vals[i*n:i*n+n]) for i in range(0, ratio)]

    return jac_vals


# def f(x, y):
#     z = x * y
#     w = z**3
#     return w + z, jnp.log(w)

# x = jnp.ones(2)
# y = 2.*jnp.ones(2)
# jaxpr = jax.make_jaxpr(jacve(f, [2, 1]))(x, y)
# print(jaxpr)

# print(jacve(f, [2, 1])(x, y))
# jacfwd_f = jacve(f, [1, 2])
# jacrev_f = jacve(f, [2, 1])
# print(timeit.timeit(lambda: jacfwd_f(x, y), number=1000))
# print(timeit.timeit(lambda: jacrev_f(x, y), number=1000))


# jac_f = jax.jacfwd(f, argnums=(0, 1))
# print(jac_f(x, y))
# print(timeit.timeit(lambda: jac_f(x, y), number=1000))

# jac_f = jax.jacrev(f, argnums=(0, 1))
# jac_f(x, y)
# print(timeit.timeit(lambda: jac_f(x, y), number=1000))


# def Helmholtz(x):
#     return x*jnp.log(x / (1. + -jnp.sum(x)))

# x = jnp.array([0.05, 0.15, 0.25, 0.35]) # jnp.ones(300)/2000. # 
# print(jax.make_jaxpr(jacve(Helmholtz, [1, 2, 3, 4, 5]))(x))

# print(jacve(Helmholtz, [1, 2, 3, 4, 5])(x))
# print(timeit.timeit(lambda: jacve(Helmholtz, [2, 5, 4, 3, 1])(x), number=1000))
# print(timeit.timeit(lambda: jacve(Helmholtz, [1, 2, 3, 4, 5])(x), number=1000))
# print(timeit.timeit(lambda: jacve(Helmholtz, [5, 4, 3, 2, 1])(x), number=1000))


# jac_Helmholtz = jax.jacfwd(Helmholtz, argnums=(0,))
# print(jax.make_jaxpr(jac_Helmholtz)(x))

# print(jac_Helmholtz(x))
# print(timeit.timeit(lambda: jac_Helmholtz(x), number=1000))


# def transpose(x, y):
#     return x.T + y

# x = jnp.ones((2, 3))
# y = jnp.ones((3, 2))
# print(jax.make_jaxpr(jacve(transpose, [1]))(x, y))
# veres = jacve(transpose, [1])(x, y)[0]
# print(veres)

# jac_matmul = jax.jacfwd(transpose, argnums=(0, 1))
# print(jax.make_jaxpr(jac_matmul)(x, y))
# revres = jac_matmul(x, y)[0]
# print(revres)

# print(jnp.allclose(veres, revres))


def matmul(x, y, z):
    return x@y + z

x = jnp.ones((2, 3))
y = jnp.ones((3,))
z = jnp.ones((2,))
print(jax.make_jaxpr(jacve(matmul, [1]))(x, y, z))
veres = jacve(matmul, [1])(x, y, z)[0]
print(veres)

jac_matmul = jax.jacfwd(matmul, argnums=(0, 1))
print(jax.make_jaxpr(jac_matmul)(x, y, z))
revres = jac_matmul(x, y, z)[0]
print(revres)

print(jnp.allclose(veres, revres))


def NeuralNetwork(x, W1, b1, W2, b2, y):
    y1 = W1 @ x
    z1 = y1 + b1
    a1 = jnp.tanh(z1)
    
    y2 = W2 @ a1
    z2 = y2 + b2
    a2 = jnp.tanh(z2)
    d = a2 - y
    return .5*jnp.sum(d**2)

x = jnp.ones(4)
y = jnp.ones(4)

W1 = jnp.ones((3, 4))
b1 = jnp.ones(3)

W2 = jnp.ones((4, 3))
b2 = jnp.ones(4)
print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))

jac_NN = jax.jacrev(NeuralNetwork, argnums=(0, 1, 2, 3, 4, 5))
revres = jac_NN(x, W1, b1, W2, b2, y)[0]
print(timeit.timeit(lambda: jac_NN(x, W1, b1, W2, b2, y), number=10000))

print(jax.make_jaxpr(jacve(NeuralNetwork, order=[9, 8, 7, 6, 5, 4, 3, 2, 1]))(x, W1, b1, W2, b2, y))
jacrev = jacve(NeuralNetwork, order=[9, 8, 7, 6, 5, 4, 3, 2, 1])
veres = jacrev(x, W1, b1, W2, b2, y)[0]
print(timeit.timeit(lambda: jacrev(x, W1, b1, W2, b2, y), number=10000))

print(jnp.allclose(veres, revres))

