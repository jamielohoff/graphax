import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from chex import Array, PRNGKey

from ..core import make_empty_edges

import jax._src.core as core


def make_random(key: PRNGKey, info: Array, p: Array = None):
    num_i, num_v, num_o = info
    edges = make_empty_edges(info)
    
    size_choices = jnp.arange(1, p.size+1)
    sizes = jrand.choice(key, size_choices, (num_v,), p=p)
    choices = jnp.arange(0, num_i)
    
    output_vertices = jrand.choice(key, jnp.arange(1, num_v), (num_o-1,), replace=False)
    output_vertices = jnp.append(output_vertices, jnp.array([num_v]))
    
    # Populate edge matrix with a fully connected graph
    for j, size in zip(jnp.arange(1, num_v+1), sizes):
        subkey, key = jrand.split(key, 2)

        if not j in output_vertices:
            choices = jnp.append(choices, jnp.array([j+num_i-1]))
        edge_positions = jrand.choice(subkey, choices, (size,), replace=False)
        def add_edge_fn(_edges, i):
            _edges = _edges.at[i, j-1].set(1.)
            return _edges, None
        edges, _ = lax.scan(add_edge_fn, edges, edge_positions)
        
    # Manage output variables
    for j in jnp.arange(1, num_v):
        lenkey, vkey, key = jrand.split(key, 3)
        if j in output_vertices:
            edges = edges.at[j+num_i-1, :].set(0.)
        else:
            if jnp.sum(edges.at[j+num_i-1, :].get()) == 0.:
                num_edges = jrand.randint(lenkey, (), 1, p.size)
                vertices = jrand.choice(vkey, jnp.arange(j, num_v), (num_edges,))
                for vertex in vertices:
                    edges = edges.at[j+num_i-1, vertex].set(1.)

    return edges


primitive_code = {}
primitive_compatibility = {}

# Compatibility rule should output modified primals, params to make 
# given primals compatible. If this is not possible then throw an error
def defcompatibility(primitive, compatibility_rule, inputs=1):
    primitive_compatibility[primitive] = (compatibility_rule, inputs)
    
defcompatibility(lax.exp_p, None)
defcompatibility(lax.log_p, None)

from functools import partial

def defcode(primitive, code_rule):
    primitive_code[primitive] = partial(code_rule, primitive.__name__)

defcompatibility(lax.sin_p, None)
defcompatibility(lax.cos_p, None)
defcompatibility(lax.tan_p, None)

defcompatibility(lax.asin_p, None)
defcompatibility(lax.acos_p, None)
defcompatibility(lax.atan_p, None)

defcompatibility(lax.sinh_p, None)
defcompatibility(lax.cosh_p, None)
defcompatibility(lax.tanh_p, None)

defcompatibility(lax.asin_p, None)
defcompatibility(lax.acosh_p, None)
defcompatibility(lax.atanh_p, None)


def bi_compatibility(*primals):
    lhs, rhs = primals

    if lhs.shape == rhs.shape:
        return primals
    elif len(lhs.shape) == len(rhs.shape):
        return False
    else:
        if len(lhs.shape) > len(rhs.shape):
            shape = list(rhs.shape)
            _shape = []
            for i, s in enumerate(lhs.shape):
                if s in shape:
                    _shape.append(s)
                    shape.pop(shape.index(s))
                else:
                    _shape.append(1)
            rhs = rhs.reshape(_shape)
        else:
            shape = list(rhs.shape)
            _shape = []
            for i, s in enumerate(lhs.shape):
                if s in shape:
                    _shape.append(s)
                    shape.pop(shape.index(s))
                else:
                    _shape.append(1)
            rhs = rhs.reshape(_shape)
        
    return lhs, rhs
    
defcompatibility(lax.add_p, bi_compatibility, inputs=2)
defcompatibility(lax.mul_p, bi_compatibility, inputs=2)
defcompatibility(lax.sub_p, bi_compatibility, inputs=2)
defcompatibility(lax.div_p, bi_compatibility, inputs=2)


def make_random_primal(key, ndims, minval=1, maxval=4):
    # Function that creates new variables for a jaxpr
    shape = None
    if ndims == 0:
        return jnp.ones(1)
    elif ndims == 1:
        shape = jrand.randint(key, (1,), minval, maxval)
    elif ndims == 2:
        shape = jrand.randint(key, (2,), minval, maxval)
    else:
        ValueError(str(ndims) + " is not a supported number of dimensions!")
        
    return jnp.ones(shape)


def sample_primitive(key):
    primitives = list(primitive_compatibility.keys())
    l = len(primitives)
    idx = jrand.randint(key, (), 0, l)
    return primitives[idx]


def make_random_eqn(key, *primals):
    # Function that creates new eqns for a jaxpr
    # TODO manage params
    prim = sample_primitive(key)
    return prim, prim.bind(*primals)


from typing import Sequence, NamedTuple, Dict
from copy import copy

class Eqn(NamedTuple):
    prim: core.Primitive
    idxs: Sequence[int]
    params: Dict = {}


def make_random_jaxpr(key: PRNGKey, info: Array):
    invars, eqns, outvars = [], [], []
    num_i, num_v, num_o = info
    
    # Create list of invars
    for _ in range(num_i):
        subkey, key = jrand.split(key, 2)
        ndims = jrand.randint(subkey, (), 0, 3)
        primal = make_random_primal(key, ndims)
        invars.append(primal)
        
    inputs = copy(invars)
    v = 0
    while v < num_v:
        pkey, vkey, key = jrand.split(key, 3)
        
        # Select primitive
        prim = sample_primitive(pkey)
        compat_fn, ninputs = primitive_compatibility[prim]            
        
        # Select inputs
        idxs = jrand.choice(vkey, jnp.array(range(len(inputs))), (ninputs,), replace=False)
        primals = [inputs[idx] for idx in idxs]
        print(primals)
        if compat_fn is not None:
            primals, params = compat_fn(*primals)
            if type(primals) is bool:
                continue
            
        val = prim.bind(*primals)
        
        # Finalize
        eqns.append(Eqn(prim, [idx for idx in idxs]))
        inputs.append(val)
        v += 1
        
    # Create list of outvars
    out_idxs = jrand.randint(vkey, (num_o-1,), len(invars), len(inputs))
    out_idxs = list(jnp.append(out_idxs, -1))
        
    def build_jaxpr(invars):
        inputs = invars
        outputs = []
        for i, eqn in enumerate(eqns):
            # Unpack Eqn first
            prim = eqn.prim
            params = eqn.params
            primals = [inputs[idx] for idx in eqn.idxs]
            
            # Apply primitive
            outvar = prim.bind(*primals, **params)
            inputs.append(outvar)
            
        # Process output variabls
        for i in out_idxs:
            outputs.append(inputs[i])
        return outputs
    
    return jax.make_jaxpr(build_jaxpr)(invars)

    
