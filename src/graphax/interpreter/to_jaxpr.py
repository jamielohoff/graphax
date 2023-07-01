from functools import wraps, partial, reduce
from collections import defaultdict

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax._src.util import safe_map
from jax._src.ad_util import Zero
import jax._src.core as core

from jax.tree_util import tree_flatten, tree_unflatten


elemental_registry = {}


def defpartial(primitive, *partialrules):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_registry[primitive] = partial(standard_partial, partialrules, primitive)


def standard_partial(partialrules, primitive, primals, **params):
    val_out = primitive.bind(*primals, **params)
    
    partials_out = [rule(*primals, **params) for rule in partialrules 
                    if rule is not None]
    return val_out, reduce(add_partials, partials_out, Zero.from_value(val_out))


# Useful for stuff such as exp_p
def defpartial2(primitive, *partialrules):
    assert isinstance(primitive, core.Primitive)
    assert not primitive.multiple_results
    elemental_registry[primitive] = partial(standard_partial2, partialrules, primitive)


def standard_partial2(partialrules, primitive, primals, **params):
    val_out = primitive.bind(*primals, **params)
    
    partials_out = [rule(val_out, *primals, **params) for rule in partialrules 
                    if rule is not None]
    return val_out, reduce(add_partials, partials_out, Zero.from_value(val_out))


def add_partials(x, y):
    if type(x) is Zero:
        return y
    elif type(y) is Zero:
        return x
    else:
        return jax._src.ad_util.add_jaxvals(x, y)
   
# TODO use multiple dispatch here! 
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
                return jnp.eye(in_size, out_size).reshape(*in_val.shape, *out_val.shape)
            elif in_size > 1 and out_size == 1:
                return jnp.ones(in_size)
            elif in_size == 1 and out_size > 1:
                return jnp.ones((1, out_size))
    return 1.


def _ones_like(val):
    if hasattr(val, "aval"):
        if type(val.aval) is core.ShapedArray:
            size = val.size
            if size > 1:
                return jnp.ones((size, size)).reshape(*val.shape, *val.shape)
    return 1.


# Manages the multiplication of Jacobians
# TODO this guy needs some serious simplification
def _mul(lhs, rhs):
    if hasattr(lhs, "aval") and hasattr(rhs, "aval"):
        if type(lhs.aval) is core.ShapedArray and type(rhs.aval) is core.ShapedArray:
            if lhs.aval.ndim == 1:
                if rhs.aval.ndim == 1:
                    return jnp.einsum("i,m->im", lhs, rhs)
                elif rhs.aval.ndim == 2:
                    if rhs.aval.shape[0] == 1:
                        return jnp.outer(lhs, rhs)
                    else:
                        return jnp.einsum("i,mn->imn", lhs, rhs)
                
            elif lhs.aval.ndim == 2:
                if rhs.aval.ndim == 1:
                    return jnp.einsum("ik,k->i", lhs, rhs)
                elif rhs.aval.ndim == 2:
                    return jnp.einsum("ik,km->im", lhs, rhs)
                elif rhs.aval.ndim == 3:
                    return jnp.einsum("ik,km->im", lhs, rhs)
                else:
                    return 0
            else:
                return jnp.einsum("ijkl,klmn->ijmn", lhs, rhs)
    return lhs * rhs
    
# Define elemental partials
defpartial(lax.neg_p, lambda x: -_eye_like1(x))
defpartial(lax.integer_pow_p, lambda x, y: y*x**(y-1))

defpartial2(lax.exp_p, lambda x: x*_eye_like1(x))
defpartial(lax.log_p, lambda x: _eye_like1(x)/x)

defpartial(lax.sin_p, lambda x: jnp.cos(x)*_eye_like1(x))
defpartial(lax.asin_p, lambda x: _eye_like1(x)/jnp.sqrt(1.0 - x**2))
defpartial(lax.cos_p, lambda x: -jnp.sin(x)*_eye_like1(x))
defpartial(lax.acos_p, lambda x: -_eye_like1(x)/jnp.sqrt(1.0 - x**2))
defpartial(lax.tan_p, lambda x: _eye_like1(x) + _eye_like1(x)*jnp.tan(x)**2)
defpartial(lax.atan_p, lambda x: _eye_like1(x)/(1.0 + x**2))

defpartial(lax.sinh_p, lambda x: _eye_like1(x)*jnp.cosh(x))
defpartial(lax.asinh_p, lambda x: _eye_like1(x)/jnp.sqrt(1.0 + x**2))
defpartial(lax.cosh_p, lambda x: _eye_like1(x)*jnp.sinh(x))
defpartial(lax.acosh_p, lambda x: _eye_like1(x)/jnp.sqrt(x**2 - 1.0))
defpartial(lax.tanh_p, lambda x: _eye_like1(x) - _eye_like1(x)*jnp.tanh(x)**2)
defpartial(lax.atanh_p, lambda x: _eye_like1(x)/(1.0 - x**2))

def add_partial(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (eye, eye)
    else:
        return (_eye_like(y, out), _eye_like(x, out))
    
def mul_partial(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (y*eye, x*eye)
    else:
        return (y*_eye_like(x, out), x*_eye_like(y, out))
        # return lambda out, x, y: (y*_eye_like(x, out), x*_eye_like(y, out))
    
def div_partial(out, x, y):
    if x.shape == y.shape:
        eye = _eye_like1(x)
        return (eye/y, -eye*x/y**2)
    else:
        return (_eye_like(x, out)/y, -_eye_like(y, out)*x/y**2)
    
defpartial2(lax.add_p, add_partial)
defpartial2(lax.mul_p, mul_partial)
defpartial2(lax.sub_p, lambda out, x, y: (_eye_like(y, out), -_eye_like(x, out)))
defpartial2(lax.div_p, div_partial)

defpartial(lax.reduce_sum_p, lambda x, axes: jnp.ones_like(x))     
defpartial(lax.transpose_p, lambda x: jnp.ones_like(x).T)        
# defpartial(lax.dot_general_p, lambda x, y: jnp.ones_like(x).T)

def jacve(fun, order, argnums=(0,)):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        # Make repackaging work properly with one input value only
        flattened_args, in_tree = tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)
        out = vertex_elimination_jaxpr(closed_jaxpr.jaxpr, order, closed_jaxpr.literals, *args, argnums=argnums)
        # out = tree_unflatten(in_tree, out)
        return out
    return wrapped


def _get_shape_dtype(val):
    if type(val) is core.ShapedArray:
        return val.shape, val.dtype
    return (), type(val)    


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

    # TODO merge the two loops into one?
    # Loop though elemental partials
    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if eqn.primitive not in elemental_registry:
            raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")
        
        primal_outvals, elemental_outvals = elemental_registry[eqn.primitive](invals, **eqn.params)
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
                edge_outval = _mul(in_val, out_val)
                if in_edge in transpose_graph[out_edge].keys():
                    _edge = transpose_graph[out_edge][in_edge]
                    edge_outval += _edge
                graph[in_edge][out_edge] = edge_outval
                transpose_graph[out_edge][in_edge] = edge_outval
                del graph[in_edge][eqn.outvars[0]]
            del transpose_graph[out_edge][eqn.outvars[0]]
        
        # Cleanup eliminated vertices            
        del graph[eqn.outvars[0]]
        del transpose_graph[eqn.outvars[0]]    

    # Collect outputs
    # TODO this can be optimized as well
    jac_vals = [graph[invar][outvar] for outvar in jaxpr.outvars for invar in jaxpr.invars]
    jacobians = [tuple(jac_vals[i:i+len(jaxpr.invars)]) for i in range(0, len(jac_vals)//len(jaxpr.outvars))]
    return jacobians

import timeit


# def f(x, y):
#     z = x * y
#     w = z**3
#     return w + z, jnp.log(w)


# x = jnp.ones(200)
# y = 2.*jnp.ones(200)
# print(jax.make_jaxpr(jacve(f, [2, 1]))(x, y))
# jacve(f, [2, 1])(x, y)

# st = time.time()
# for _ in range(1000):
#     jacve(f, [2, 1], argnums=(0, 1))(x, y)
# print(time.time() - st)

# jac_f = jax.jacfwd(f, argnums=(0,))
# jac_f(x, y)

# st = time.time()
# for _ in range(1000):
#     jac_f(x, y)
# print(time.time() - st)


def Helmholtz(x):
    z = jnp.log(x / (1. + -jnp.sum(x)))
    return x * z


x = jnp.ones(400)/500.
print(jax.make_jaxpr(jacve(Helmholtz, [1, 2, 3, 4, 5]))(x))

print(jacve(Helmholtz, [1, 2, 3, 4, 5])(x))
print(timeit.timeit(lambda: jacve(Helmholtz, [2, 5, 4, 3, 1])(x), number=1000))
print(timeit.timeit(lambda: jacve(Helmholtz, [1, 2, 3, 4, 5])(x), number=1000))
print(timeit.timeit(lambda: jacve(Helmholtz, [5, 4, 3, 2, 1])(x), number=1000))


jac_Helmholtz = jax.jacfwd(Helmholtz, argnums=(0,))
print(jax.make_jaxpr(jac_Helmholtz)(x))

print(jac_Helmholtz(x))
print(timeit.timeit(lambda: jac_Helmholtz(x), number=1000))


# def matmul(x, y):
#     z = x @ y
#     w = x.T + y
#     return z, w

# x = jnp.ones((2, 3))
# y = jnp.ones((3, 2))
# print(jax.make_jaxpr(vertex_elimination_jacobian(matmul, [3, 2, 1]))(x, y))
# print("Result:", vertex_elimination_jacobian(matmul, [3, 2, 1])(x, y))

# jac_matmul = jax.jacfwd(matmul, argnums=(0, 1))
# print(jax.make_jaxpr(jac_matmul)(x, y))
# print(jac_matmul(x, y))


# def NeuralNetwork(x, W1, b1, W2, b2, y):
#     y1 = W1 @ x
#     z1 = y1 + b1
#     a1 = jnp.tanh(z1)
    
#     y2 = W2 @ a1
#     z2 = y2 + b2
#     a2 = jnp.tanh(z2)
#     d = a2 - y
#     e = d**2
#     return .5*jnp.sum(e)

# x = jnp.ones(4)
# y = jnp.ones(4)

# W1 = jnp.ones((3, 4))
# b1 = jnp.ones(3)

# W2 = jnp.ones((4, 3))
# b2 = jnp.ones(4)
# print(jax.make_jaxpr(NeuralNetwork)(x, W1, b1, W2, b2, y))
# jac_Helmholtz = jax.jacrev(Helmholtz, argnums=(1,))
# print(jax.make_jaxpr(jac_Helmholtz)(x))
# print(jac_Helmholtz(x))

