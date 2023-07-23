from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.lax as lax
import jax.random as jrand

from jax._src.util import safe_map

jaxpr = None
primitive_code = {}

mono_prims = set()
bi_prims = set()
reduce_prims = set()
special_prims = set()
dot_general_prims = set()


def defmonocode(primitive, code_rule, ninputs=1):
    mono_prims.add(primitive)
    rule = partial(code_rule, primitive)
    primitive_code[primitive] = (rule, ninputs)
    

def defbicode(primitive, code_rule, ninputs=1):
    bi_prims.add(primitive)
    rule = partial(code_rule, primitive)
    primitive_code[primitive] = (rule, ninputs)
    

def defreducecode(primitive, code_rule, ninputs=1):
    reduce_prims.add(primitive)
    rule = partial(code_rule, primitive)
    primitive_code[primitive] = (rule, ninputs)    
    

def defspecialcode(primitive, code_rule, ninputs=1):
    special_prims.add(primitive)
    rule = partial(code_rule, primitive)
    primitive_code[primitive] = (rule, ninputs)
    

def defdotcode(primitive, code_rule, ninputs=1):
    dot_general_prims.add(primitive)
    rule = partial(code_rule, primitive)
    primitive_code[primitive] = (rule, ninputs)

    
def monorule(primitive, key, outvars, invars, invals):
    name = primitive.__name__
    val = primitive(*invals)
    return val, f"{outvars} = jnp.{name}({invars[0]})\n"
    
defmonocode(jnp.negative, monorule)
defmonocode(jnp.exp, monorule)
defmonocode(jnp.log, monorule)

defmonocode(jnp.sin, monorule)
defmonocode(jnp.cos, monorule)
defmonocode(jnp.tan, monorule)

defmonocode(jnp.arcsin, monorule)
defmonocode(jnp.arccos, monorule)
defmonocode(jnp.arctan, monorule)

defmonocode(jnp.sinh, monorule)
defmonocode(jnp.cosh, monorule)
defmonocode(jnp.tanh, monorule)

defmonocode(jnp.arcsinh, monorule)
defmonocode(jnp.arctanh, monorule)

defmonocode(jnp.square, monorule)
defmonocode(jnp.sqrt, monorule)


def lax_monorule(primitive, key, outvars, invars, invals):
    name = primitive.__name__
    val = primitive(*invals)
    return val, f"{outvars} = lax.{name}({invars[0]})\n"

defspecialcode(lax.stop_gradient, lax_monorule)
defspecialcode(jnp.squeeze, monorule)


def integer_pow_rule(primitive, key, outvars, invars, invals):
    name = primitive.__name__
    exponent = int(jrand.randint(key, (), 2, 5))
    val = primitive(*invals, exponent)
    return val, f"{outvars} = jnp.{name}({invars[0]},{exponent})\n"

defmonocode(jnp.power, integer_pow_rule)


def reduce_rule(primitive, key, outvars, invars, invals):
    name = primitive.__name__
    primal = invals[0]
    if len(primal.shape) != 0:
        size = primal.ndim
        axis = int(jrand.randint(key, (), 0, size))
        params = {"axis": axis}
        lines = f"{outvars} = jnp.{name}({invars[0]}, axis={axis})\n"
        val = primitive(*invals, **params)
        return val, lines
    else:
        return None, ""

defreducecode(jnp.sum, reduce_rule)
defreducecode(jnp.max, reduce_rule)
defreducecode(jnp.min, reduce_rule)
defreducecode(jnp.prod, reduce_rule)


# TODO include automatic broadcasting?
def birule(primitive, key, outvar, invars, invals):
    assert len(invars) == 2
    name = primitive.__name__
    lhs, rhs = invals
    compatible = True
    lines = ""
    val = None
    if len(lhs.shape) != 0 and len(rhs.shape) != 0:
        if len(lhs.shape) == len(rhs.shape):
            compatible = all([lhs.shape == rhs.shape])
        elif len(lhs.shape) > len(rhs.shape):
            shape = list(rhs.shape)
            _shape = []
            for s in lhs.shape:
                if s in shape:
                    shape.pop(shape.index(s))
                    _shape.append(s)
                else:
                    _shape.append(1)
            compatible = len(shape) == 0
            if compatible:
                lines += f"_{invars[1]} = {invars[1]}.reshape({_shape})\n"
                lines += "    "
                invars[1] = f"_{invars[1]}"
                invals[1] = jnp.reshape(invals[1], _shape)
                val = primitive(*invals)
        elif len(lhs.shape) < len(rhs.shape):
            shape = list(lhs.shape)
            _shape = []
            for s in rhs.shape:
                if s in shape:
                    shape.pop(shape.index(s))
                    _shape.append(s)
                else:
                    _shape.append(1)
            compatible = len(shape) == 0
            if compatible:
                lines += f"_{invars[0]} = jnp.reshape({invars[0]}, {_shape})\n"
                lines += "    "
                invars[0] = f"_{invars[0]}"
                invals[0] = jnp.reshape(invals[0], _shape)
                val = primitive(*invals)
    else:
        val = primitive(*invals)
    lines += f"{outvar} = jnp.{name}(" + ",".join(invars) + ")\n"
    
    return val, lines

defbicode(jnp.add, birule, ninputs=2)
defbicode(jnp.multiply, birule, ninputs=2)
defbicode(jnp.divide, birule, ninputs=2)
defbicode(jnp.subtract, birule, ninputs=2)


def transpose_rule(primitive, key, outvar, invars, invals):   
    name = primitive.__name__
    primal = invals[0] 
    if len(primal.shape) != 0:
        axes = jnp.array([i for i in range(primal.ndim)])
        perm = jrand.permutation(key, axes)
        perm = [int(p) for p in perm]
        
        params = {"axes":perm}
        lines = f"{outvar} = jnp.{name}({invars[0]},axes={perm})\n"
        val = primitive(*invals, **params)
        return val, lines
    else:
        return None, ""
    
defspecialcode(jnp.transpose, transpose_rule)


def dot_general_rule(primitive, key, outvar, invars, invals):
    assert len(invals) == 2
    name = primitive.__name__
    lhs, rhs = invals

    if len(lhs.shape) != 0 and len(rhs.shape) != 0:
        dim_pairs = []
        for i, s in enumerate(lhs.shape):
            for j, t in enumerate(rhs.shape):
                if s == t:
                    dim_pairs.append((i, j))
        if len(dim_pairs) == 0:
            return None, ""
        else:
            idx = jrand.randint(key, (), 0, len(dim_pairs))
            pair = dim_pairs[idx]
            dim_nums = (((pair[0],), (pair[1],)), ((), ()))
            params = {"dimension_numbers":dim_nums}
            val = primitive(*invals, **params)
            lines = f"{outvar} = lax.{name}(" + ",".join(invars) + f",dimension_numbers={dim_nums})\n"
            return val, lines
    else:
        return None, ""
    
defdotcode(lax.dot_general, dot_general_rule, ninputs=2)


def slice_rule(primitive, key, outvar, invars, invals):
    name = primitive.__name__
    primal = invals[0]
    if len(primal.shape) != 0:
        start_indices = []
        limit_indices = []
        for s in primal.shape:
            idxs = jrand.randint(key, (2,), 0, s).sort()
            idxs = [int(i) for i in idxs]
            start_indices.append(idxs[0])
            if idxs[1] == 0:
                limit_indices.append(idxs[1]+1)
            else:
                limit_indices.append(idxs[1]+1)
        params = {"start_indices":start_indices, "limit_indices":limit_indices}
        lines = f"{outvar} = lax.{name}({invars[0]},start_indices={start_indices}, limit_indices={limit_indices})\n"
        val = primitive(*invals, **params)
        return val, lines
    else:
        return None, ""
    
defspecialcode(lax.slice, slice_rule)


def broadcast_rule(primitive, key, outvar, invars, invals):
    name = primitive.__name__
    primal = invals[0]
    if len(primal.shape) < 2:
        shape = list(primal.shape)
        idx = jrand.randint(key, (), 0, len(primal.shape)+1)
        shape.insert(idx, 1)
        params = {"newshape":shape}
        lines = f"{outvar} = jnp.{name}({invars[0]},newshape={shape})\n"
        val = primitive(*invals, **params)
        return val, lines
    else:
        return None, ""
    
defspecialcode(jnp.reshape, broadcast_rule)


def sample_primitive(key, prim_p):
    subkey, key = jrand.split(key, 2)
    N = jrand.choice(key, jnp.arange(0, 5), p=prim_p)
    primitives = mono_prims
    if N == 0:
        primitives = list(mono_prims)
    elif N == 1:
        primitives = list(bi_prims)
    elif N == 2:
        primitives = list(reduce_prims)
    elif N == 3:
        primitives = list(special_prims)
    elif N == 4:
        primitives = list(dot_general_prims)
        
    l = len(primitives)
    idx = jrand.randint(subkey, (), 0, l)
    return primitives[idx]


def make_random_primal(key, ndims, minval=1, maxval=5):
    # Function that creates new variables for a jaxpr
    shape = None
    subkey, key = jrand.split(key, 2)
    if ndims == 0:
        return jrand.uniform(subkey, ())
    elif ndims == 1:
        shape = jrand.randint(key, (1,), 2, maxval)
    elif ndims == 2:
        shape = jrand.randint(key, (2,), minval, maxval)
    else:
        ValueError(str(ndims) + " is not a supported number of dimensions!")
        
    return jrand.uniform(subkey, shape)


jaxpr = None
def make_random_code(key, 
                    info, 
                    primal_p=jnp.array([.15, .5, .35]),
                    prim_p=jnp.array([.15, .35, .06, .12, .31]),
                    max_literals=3):
    code = "import jax\nimport jax.numpy as jnp\n"
    num_i, num_v, num_o = info
    env = {}
    count = 0
    
    def read(var: str):
        return env[var]
    
    def write(var: str, val: Any):
        env[var] = val
        
     # Create list of invars
    for _ in range(num_i):
        subkey, key = jrand.split(key, 2)
        ndims = jrand.choice(subkey, jnp.arange(0, 3), (), p=primal_p)
        primal = make_random_primal(key, ndims)
        var = "v"+str(count)
        write(var, primal)
        count += 1

    f_inputs = list(env.keys())
    function_header = "def f(" + ",".join(f_inputs) + "):\n"
    code += function_header
    
    # Add literals
    num_literals = jrand.randint(key, (), 0, max_literals) # number of literals
    for _ in range(num_literals):
        subkey, key = jrand.split(key, 2)
        literal = make_random_primal(subkey, ndims)
        shape = literal.shape
        var = "v"+str(count)
        code += f"    {var} = jnp.ones({shape})\n"
        write(var, literal)
        count += 1
        
    # Adding randomly chosen equations
    v = 0
    inputs = list(env.keys())
    inputs_p = [1.]*len(inputs)
    while v < num_v:
        pkey, rkey, vkey, key = jrand.split(key, 4)
        
        # Select primitive
        prim = sample_primitive(pkey, prim_p=prim_p)
        rule, num_inputs = primitive_code[prim]            
        
        # Select inputs
        _p = jnp.array(inputs_p)
        p = _p/_p.sum()
        idxs = jrand.choice(vkey, jnp.array(range(len(inputs))), (num_inputs,), replace=False, p=p)
        vars = [inputs[idx] for idx in idxs]
        primals = safe_map(read, vars)

        var = "v" + str(count)
        val, lines = rule(rkey, var, vars, primals)
        # Add source code
        if val is not None:
            write(var, val)
            code += f"    " + lines
            count += 1
            v += 1
            for idx in idxs:
                inputs_p[idx] *= .15
            inputs.append(var)
            inputs_p.append(1.)
        
    # Create list of outvars
    inputs = list(env.keys())
    out_idxs = jrand.choice(vkey, jnp.arange(num_i+num_literals, len(inputs)), (num_o-1,), replace=False)
    out_idxs = list(jnp.append(out_idxs, -1))
    outputs = [inputs[idx] for idx in out_idxs]

    code += "    return " + ",".join(outputs) + "\n"
    code += "global jaxpr\n"
    code += "jaxpr = jax.make_jaxpr(f)(" + ",".join(f_inputs) + ")"

    input_params = {v:env[v] for v in f_inputs}
    exec(code, globals(), input_params)
    
    return code, jaxpr  

