from typing import Callable, Tuple, Union, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import Var, ClosedJaxpr, JaxprEqn

from chex import Array

from graphax.core import GraphInfo, make_graph_info
    
# Needed to resolve PJIT subgraphs
PJIT = {jax._src.pjit.pjit_p}
# Reshaping of tensors. Does not change the Jacobian accumulation as slicing also
# merely copies the respective partials. However, it terminates the derivative
# flow over "sliced-off" edges.
RESHAPING = {lax.concatenate_p, lax.squeeze_p, lax.convert_element_type_p,
            lax.slice_p, lax.dynamic_slice_p, lax.dynamic_update_slice_p}
# Parallel operations
PARALLEL = {lax.add_p, lax.mul_p, lax.sub_p, lax.div_p, lax.log_p, lax.sin_p, 
            lax.cos_p, lax.tan_p, lax.asin_p, lax.acos_p, lax.atan_p,
            lax.sinh_p, lax.cosh_p, lax.tanh_p, lax.asinh_p, lax.acosh_p,
            lax.atanh_p, lax.exp_p, lax.integer_pow_p, lax.neg_p, lax.eq_p,
            lax.gt_p, lax.ge_p, lax.lt_p, lax.le_p}
# Accumulation operations
ACCUMULATION = {lax.reduce_sum_p, lax.reduce_prod_p, lax.reduce_max_p, 
                lax.reduce_min_p, lax.dot_general_p}
# Broadcasting operations
BROADCASTING = {lax.broadcast_in_dim_p}
    

def filter_eqns(eqns: Sequence[JaxprEqn]) -> Sequence[JaxprEqn]:
    """
    Function that filters out assignments of unused variables.
    """
    return [eqn for eqn in eqns if not str(eqn.outvars[0]) == "_"]


def make_graph(f_jaxpr: Union[ClosedJaxpr, Callable], *xs: Array) -> Tuple[Array, GraphInfo, Array, Array]:
    """
    Function that creates a computational graph from a JAX input function or a jaxpr.
    """
    jaxpr = jax.make_jaxpr(f_jaxpr)(*xs) if isinstance(f_jaxpr, Callable) else f_jaxpr
            
    num_i = len(jaxpr.jaxpr._invars)
    num_o = len(jaxpr.jaxpr._outvars)
    eqns = filter_eqns(jaxpr.eqns)
    num_v = len(eqns)
       
    info = make_graph_info([num_i, num_v, num_o])
    edges = jnp.zeros((3, num_i+num_v+1, num_v))  
    
    is_invar_list = []
    
    # Processing input variables
    variables = {}
    counter = 0
    for invar in jaxpr.jaxpr._invars:
        variables[str(invar)] = counter
        counter += 1

    # Process intermediate variables
    i = 0
    for eqn in eqns:
        for outvar in eqn.outvars:
            if str(outvar) not in variables:
                variables[str(outvar)] = counter
                counter += 1
            j = 0
            for invar in eqn.invars:
                if invar in jaxpr.jaxpr._outvars:
                    is_invar_list.append(invar)
                if isinstance(invar, Var):
                    if eqn.primitive in PJIT:
                        # Needs special treatment
                        continue

                    elif eqn.primitive in RESHAPING:
                        # Slicing operation is a parallel operation that selects
                        # values from an array and by virtue of this the flow of
                        # partials along the respective edges
                        j = variables[str(invar)]
                        # Set operation type
                        edges = edges.at[0, 0, i].set(4.)
                        # Set sparsity type 0 because no fmas are done
                        structure = jnp.array([0., invar.aval.size, outvar.aval.size])
                        edges = edges.at[:, j+1, i].set(structure)
                        
                    elif eqn.primitive in PARALLEL:
                        j = variables[str(invar)]
                        # Set operation type
                        edges = edges.at[0, 0, i].set(1.)
                        # Set sparsity type
                        if invar.aval.size == 1 and outvar.aval.size == 1:
                            # Singleton operations are not sparse
                            structure = jnp.array([1., invar.aval.size, outvar.aval.size])
                            edges = edges.at[:, j+1, i].set(structure) 
                        else:
                            # Parallel operations are sparse
                            structure = jnp.array([2., invar.aval.size, outvar.aval.size])
                            edges = edges.at[:, j+1, i].set(structure) 
                                          
                    elif eqn.primitive in ACCUMULATION:
                        # Accumulation operation
                        j = variables[str(invar)]
                        # Set operation type
                        edges = edges.at[0, 0, i].set(2.)
                        # Set sparsity type
                        structure = jnp.array([1., invar.aval.size, outvar.aval.size])
                        edges = edges.at[:, j+1, i].set(structure)  
                        
                    elif eqn.primitive in BROADCASTING:
                        # Accumulation operation
                        j = variables[str(invar)]
                        # Set operation type
                        edges = edges.at[0, 0, i].set(3.)
                        # Set sparsity type
                        structure = jnp.array([1., invar.aval.size, outvar.aval.size])
                        edges = edges.at[:, j+1, i].set(structure)  
                        
                    else:
                        print("Primitive not supported")
                    j += 1
            i += 1
                        
    # Processing output variables
    vertex_mask = jnp.zeros(num_v)
    k = 0
    for outvar in jaxpr.jaxpr._outvars:
        if not outvar in is_invar_list:
            idx = variables[str(outvar)]
            vertex_mask = vertex_mask.at[k].set(idx)
            k += 1
                
    # Make attention mask
    attn_mask = jnp.ones((num_v, num_v))
    return edges, info, vertex_mask, attn_mask


def get_fmas_jacprod(_jac_edges, fmas, col, _col, nonzero, vertex, info):
    # Define aliases
    num_i = info.num_inputs
    col_sparsity = col.at[0, :].get()
    col_ins = col.at[1, :].get()
    col_outs = col.at[2, :].get()
    
    _col_sparsity = _col.at[0, :].get()
    _col_ins = _col.at[1, :].get()
    _col_outs = _col.at[2, :].get()
    
    # Calculate fmas
    in_div = jnp.where(col_sparsity > 1., 1./col_ins, 1.)
    in_div = jnp.where(col_sparsity > 0., in_div, 0.)
    
    out_div = lax.cond(_col_sparsity[vertex+num_i-1] > 1., 
                        lambda: 1./_col_ins[vertex+num_i-1],
                        lambda: 1.)
    out_div = lax.cond(_col_sparsity[vertex+num_i-1] > 0., 
                        lambda: out_div,
                        lambda: 0.)

    _fmas = col_ins*col_outs*_col_outs[vertex+num_i-1]*out_div
    fmas = jnp.sum(_fmas*in_div)
    
    # Adjust columns 
    # Sparsity Column  
    # Sparsity structure 1: dense Jacobian
    # Sparsity structure 2: diagonal Jacobian
    new_sparsity_col = col_sparsity + _col_sparsity
    new_sparsity_col = jnp.where(new_sparsity_col == 3., 1., new_sparsity_col) # sparse*dense or dense*sparse ops get dense
    new_sparsity_col = jnp.where(new_sparsity_col == 4., 2., new_sparsity_col) # sparse*sparse ops stay sparse
    flag = jnp.logical_and(new_sparsity_col == 0., col_ins > 0.)
    new_sparsity_col = jnp.where(flag, _col_sparsity[vertex+num_i-1], new_sparsity_col)

    # In shape column
    new_in_col = jnp.where(col_ins > 0., col_ins, _col_ins)
    
    # Out shape column
    new_out_col = jnp.where(_col_outs > 0., _col_outs, 0.)
    new_out_col = jnp.where(col_outs > 0., _col_outs[vertex+num_i-1], new_out_col)
        
    # Set the Jacobian
    _jac_edges = _jac_edges.at[0, :, nonzero].set(new_sparsity_col)
    _jac_edges = _jac_edges.at[1, :, nonzero].set(new_in_col)
    _jac_edges = _jac_edges.at[2, :, nonzero].set(new_out_col)
            
    return _jac_edges, fmas


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def vectorized_vertex_eliminate(vertex: int, mat: Array, info: GraphInfo) -> Tuple[Array, float]:
    """
    Fully JIT-compilable function that implements the vertex-elimination procedure. 
    Vertex elimination means that we front-eliminate all incoming edges and 
    back-eliminate all outgoing edges of a given vertex. However, the implementation
    here does not make use of the function above to be more efficient.

    Arguments:
        vertex (int): Vertex we want to eliminate.
        edges (Array): Matrix that describes the connectivity of the 
                        computational graph.
        info (GraphInfo): Meta-information about the computational graph.

    Returns:
        A tuple that contains the new edge representation of the computational
        graph and the number of fmas (fused multiplication-addition ops).
    """
    num_inputs = info.num_inputs
    num_intermediates = info.num_intermediates
    jac_edges = mat.at[:, 1:, :].get()
    col = jac_edges.at[:, :, vertex-1].get()
        
    def update_edges_fn(carry, nonzero):
        _jac_edges, fmas = carry
        # Get the index of the operation and the 
        _col = _jac_edges.at[:, :, nonzero].get()
        
        # Calculate the fma operations and the new shapes of the Jacobians for 
        # the respective and update the vertex
        _jac_edges, _fmas = lax.cond(nonzero > -1, 
                                lambda _e, f, c, _c, nz, v: get_fmas_jacprod(_e, f, c, _c, nz, v, info), 
                                lambda _e, f, c, _c, nz, v: (_e, 0.), 
                                _jac_edges, fmas, col, _col, nonzero, vertex)
        fmas += _fmas        
        carry = (_jac_edges, fmas)
        return carry, 0
    
    nonzeros = jnp.nonzero(jac_edges.at[0, num_inputs+vertex-1, :].get(),
                           size=num_intermediates,
                           fill_value=-1)[0].T
    
    output, _ = lax.scan(update_edges_fn, (jac_edges, 0.), nonzeros)
    jac_edges, fmas = output
    jac_edges = jac_edges.at[:, num_inputs+vertex-1, :].set(0)
    jac_edges = jac_edges.at[:, :, vertex-1].set(0)

    mat = mat.at[:, 1:, :].set(jac_edges)
    return mat, fmas


def Helmholtz(x):
    e = jnp.sum(x)
    f = 1. + -e
    w = x / f
    z = jnp.log(w)
    return x*z

x = jnp.ones(4)
print(jax.make_jaxpr(Helmholtz)(x))
edges, info, vertex_mask, attn_mask = make_graph(Helmholtz, x)
# print(edges)

edges, fmas = vectorized_vertex_eliminate(2, edges, info)
# print(edges, fmas)

edges, _fmas = vectorized_vertex_eliminate(5, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(4, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(3, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(1, edges, info)
fmas += _fmas
# print(edges, _fmas)
print(fmas)

def f(x, y):
    x = x[1:]
    y = y[:-1]
    z = x * y
    w = jnp.sin(z)
    a = jnp.concatenate((z + w, jnp.zeros(2)))
    return a, jnp.log(w)

x = jnp.ones(4)
y = jnp.ones(4)
print(jax.make_jaxpr(f)(x, y))
edges, info, vertex_mask, attn_mask = make_graph(f, x, y)
print(edges)

edges, fmas = vectorized_vertex_eliminate(1, edges, info)
# print(edges, fmas)

edges, _fmas = vectorized_vertex_eliminate(2, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(3, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(4, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(5, edges, info)
fmas += _fmas
# print(edges, _fmas)

edges, _fmas = vectorized_vertex_eliminate(6, edges, info)
fmas += _fmas
# print(edges, _fmas)
print(fmas)

