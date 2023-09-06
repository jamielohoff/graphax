import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp

from jax._src.core import Var


vertex_registry = {}


def get_shape(var: Var):
    """
    Returns the appropriate shape of a singleton, vector or matrix.
    Singletons are treated as tensors with shape (1, 1)
    Row- and column vectors are treated as tensors with shape (1, n) and (n, 1)
    Matrices are treated as tensors of shape (n, m)
    """
    var_shape = jnp.array(var.aval.shape)
    if var.aval.size == 1:
        var_shape = jnp.array([1, 1])
    if len(var.aval.shape) == 1:
        var_shape = jnp.array([var.aval.shape[0], 1])
    return var_shape


def filter_invars(eqn, variables):
    filtered = [invar for invar in eqn.invars if isinstance(invar, Var)]
    return [invar for invar in filtered if variables[str(invar)] != -1]


def add_mono_vertex(edges, eqn, variables):
    """
    Adds a new vertex that corresponds to a functions with one input and one output.
    """
    filtered_invars = filter_invars(eqn, variables)

    if len(filtered_invars) == 0:
        return edges
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    # Input is singleton
    if _invar_shape[0] == 1 and _invar_shape[1] == 1:
        sparsity_type = 1
    # Input is column-vector
    elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
        sparsity_type = 2
    # Input is row-vector
    elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
        sparsity_type = 3
    # Input is matrix
    else:
        sparsity_type = 6
        
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(eqn.invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    edges = edges.at[0, i, j].set(sparsity_type)
    structure = jnp.concatenate([_invar_shape, _outvar_shape]) 
    edges = edges.at[1:, i, j].set(structure)
    
    return edges

vertex_registry[lax.neg_p] = add_mono_vertex
vertex_registry[lax.abs_p] = add_mono_vertex

vertex_registry[lax.exp_p] = add_mono_vertex
vertex_registry[lax.log_p] = add_mono_vertex

vertex_registry[lax.sin_p] = add_mono_vertex
vertex_registry[lax.cos_p] = add_mono_vertex
vertex_registry[lax.tan_p] = add_mono_vertex

vertex_registry[lax.asin_p] = add_mono_vertex
vertex_registry[lax.acos_p] = add_mono_vertex
vertex_registry[lax.atan_p] = add_mono_vertex
vertex_registry[lax.atan2_p] = add_mono_vertex

vertex_registry[lax.sinh_p] = add_mono_vertex
vertex_registry[lax.cosh_p] = add_mono_vertex
vertex_registry[lax.tanh_p] = add_mono_vertex

vertex_registry[lax.asinh_p] = add_mono_vertex
vertex_registry[lax.acosh_p] = add_mono_vertex
vertex_registry[lax.atanh_p] = add_mono_vertex

vertex_registry[lax.integer_pow_p] = add_mono_vertex
vertex_registry[lax.sqrt_p] = add_mono_vertex
vertex_registry[lax.rsqrt_p] = add_mono_vertex
vertex_registry[lax.logistic_p] = add_mono_vertex

# We currently included the custom derivative operator here 
# to enable spiking functions
vertex_registry[jax._src.custom_derivatives.custom_jvp_call_p] = add_mono_vertex


def add_bi_vertex(edges, eqn, variables):
    """
    Adds a vertex for a function with two inputs and one output. Also handles
    the broadcasting for different input shapes.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    for invar in filtered_invars:
        _invar_shape = get_shape(invar)
        _outvar_shape = get_shape(eqn.outvars[0])
    
        # Input is singleton
        if _invar_shape[0] == 1 and _invar_shape[1] == 1:
            # Output is singleton
            if _outvar_shape[0] == 1 and _outvar_shape[1] == 1:
                sparsity_type = 1
            # Output is column-vector
            elif _outvar_shape[0] > 1 and _outvar_shape[1] == 1:
                sparsity_type = 1
            # Output is row-vector
            elif _outvar_shape[0] == 1 and _outvar_shape[1] > 1:
                sparsity_type = 1
            # Output is matrix
            else:
                sparsity_type = 1.
                
        # Input is column-vector 
        elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
            # Output is column-vector
            if _outvar_shape[0] > 1 and _outvar_shape[1] == 1:
                sparsity_type = 2
            # Output is matrix, e.g. outer product
            else:
                sparsity_type = 6
                
        # Input is row-vector      
        elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
            # Output is row-vector
            if _outvar_shape[0] == 1 and _outvar_shape[1] > 1:
                sparsity_type = 3
            # Output is matrix, e.g. outer product
            else:
                sparsity_type = 6
                
        # Input is matrix
        else:
            sparsity_type = 6.
            
        num_i = edges.at[0, 0, 0].get()
        i = variables[str(invar)]
        j = variables[str(eqn.outvars[0])] - num_i - 1
        
        edges = edges.at[0, i, j].set(sparsity_type)
        
        structure = jnp.concatenate([_invar_shape, _outvar_shape]) 
        edges = edges.at[1:, i, j].set(structure)
    return edges

vertex_registry[lax.add_p] = add_bi_vertex
vertex_registry[lax.atan2_p] = add_bi_vertex
vertex_registry[lax.mul_p] = add_bi_vertex
vertex_registry[lax.sub_p] = add_bi_vertex
vertex_registry[lax.div_p] = add_bi_vertex
vertex_registry[jax._src.ad_util.add_any_p] = add_bi_vertex
vertex_registry[jax.ad.add_jaxvals_p] = add_bi_vertex
vertex_registry[lax.eq_p] = add_bi_vertex
vertex_registry[lax.pow_p] = add_bi_vertex


def add_dot_general_vertex(edges, eqn, variables):
    """
    Adds a vertex that corresponds to the XLA dot_general primitive. 
    Dot general contains matrix-vector, vector-matrix, matrix-matrix and 
    dot products as a subset.
    """
    _invar_shape_0 = get_shape(eqn.invars[0])
    _invar_shape_1 = get_shape(eqn.invars[1])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    sparsity_type_0 = 1
    sparsity_type_1 = 1
    
    # Input 0 is singleton
    if _invar_shape_0[0] == 1 and _invar_shape_0[1] == 1:
        # Input 1 is singleton
        if _invar_shape_1[0] == 1 and _invar_shape_1[1] == 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 1
        # Input 1 is row-vector
        elif _invar_shape_1[0] == 1 and _invar_shape_1[1] > 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 1
            
    # Input 0 is column-vector
    elif _invar_shape_0[0] > 0 and _invar_shape_0[1] == 1:
        # Input 1 is singleton
        if _invar_shape_1[0] == 1 and _invar_shape_1[1] == 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 1
        # Input 1 is column-vector, i.e. dot product
        elif _invar_shape_1[0] > 1 and _invar_shape_1[1] == 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 1
    
    # Input 0 is row-vector
    elif _invar_shape_0[0] == 1 and _invar_shape_0[1] > 1:                   
        # Input 1 is column_vector
        if _invar_shape_1[0] == 1 and _invar_shape_1[1] == 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 3
        # Input 1 is matrix
        elif _invar_shape_1[0] > 1 and _invar_shape_1[1] > 1:
            sparsity_type_0 = 1
            sparsity_type_1 = 3
        
    # Input 0 is matrix
    else:
        # Input 1 is column-vector
        if _invar_shape_1[0] > 1 and _invar_shape_1[1] == 1:
            sparsity_type_0 = 2
            sparsity_type_1 = 1
        # Input 1 is matrix
        elif _invar_shape_1[0] > 1 and _invar_shape_1[1] > 1:
            sparsity_type_0 = 2
            sparsity_type_1 = 3
            
    # Treat Literals and Vars appropriately
    num_i = edges.at[0, 0, 0].get()
    j = variables[str(eqn.outvars[0])] - num_i - 1
    # Only first variable is a Var
    if isinstance(eqn.invars[0], Var) and not isinstance(eqn.invars[1], Var):
        i0 = variables[str(eqn.invars[0])]
        
        edges = edges.at[0, i0, j].set(sparsity_type_0)
        structure = jnp.concatenate([_invar_shape_0, _outvar_shape]) 
        edges = edges.at[1:, i0, j].set(structure)
    # Only second variable is a Var
    elif not isinstance(eqn.invars[0], Var) and isinstance(eqn.invars[1], Var):
        i1 = variables[str(eqn.invars[1])]
        
        edges = edges.at[0, i1, j].set(sparsity_type_1)
        structure = jnp.concatenate([_invar_shape_1, _outvar_shape]) 
        edges = edges.at[1:, i1, j].set(structure)
    # Both variables are Var
    else:
        i0 = variables[str(eqn.invars[0])]
        i1 = variables[str(eqn.invars[1])]
        edges = edges.at[0, i0, j].set(sparsity_type_0)
        structure_0 = jnp.concatenate([_invar_shape_0, _outvar_shape]) 
        edges = edges.at[1:, i0, j].set(structure_0)
        
        edges = edges.at[0, i1, j].set(sparsity_type_1)
        structure_1 = jnp.concatenate([_invar_shape_1, _outvar_shape]) 
        edges = edges.at[1:, i1, j].set(structure_1)
    return edges
    
vertex_registry[lax.dot_general_p] = add_dot_general_vertex
    

def add_accumulation_vertex(edges, eqn, variables):
    """
    Adds a vertex for an accumulation function.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    # Input is singleton or row/column vector
    sparsity_type = 1
    
    # Input is matrix
    if _invar_shape[0] > 1 and _invar_shape[1] > 1: 
         # Output is number, i.e. all elements are summed
        if _outvar_shape[0] == 1 and _outvar_shape[1] == 1:
            sparsity_type = 1
        # Output is column-vector, i.e. summing over rows
        elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
            sparsity_type = 2
        # Output is row-vector, i.e. summing over columns
        elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
            sparsity_type = 3
    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    edges = edges.at[0, i, j].set(sparsity_type)
    structure = jnp.concatenate([_invar_shape, _outvar_shape]) 
    edges = edges.at[1:, i, j].set(structure)
    return edges

vertex_registry[lax.reduce_sum_p] = add_accumulation_vertex
vertex_registry[lax.reduce_prod_p] = add_accumulation_vertex
vertex_registry[lax.reduce_max_p] = add_accumulation_vertex
vertex_registry[lax.reduce_min_p] = add_accumulation_vertex


def add_transpose_vertex(edges, eqn, variables):
    """
    Adds a vertex for  vector or matrix transpose operation.
    """
    filtered_invars = filter_invars(eqn, variables)
    
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    # Input is singleton
    if _invar_shape[0] == 1 and _invar_shape[1] == 1: 
        sparsity_type = 1
    
    # Input is column-vector
    elif _invar_shape[0] > 1 and _invar_shape[1] == 1:
        sparsity_type = 4
    
    # Input is row-vector
    elif _invar_shape[0] == 1 and _invar_shape[1] > 1:
        sparsity_type = 5
        
    # Input is matrix
    else:
        sparsity_type = 7
            
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    edges = edges.at[0, i, j].set(sparsity_type)
    structure = jnp.concatenate([_invar_shape, _outvar_shape]) 
    edges = edges.at[1:, i, j].set(structure)
    return edges

vertex_registry[lax.transpose_p] = add_transpose_vertex


def add_stop_gradient_vertex(edges, eqn, variables):
    """
    Adds a vertex a stop_gradient operation.
    """
    filtered_invars = filter_invars(eqn, variables)
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    sparsity_type = 0
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    edges = edges.at[0, i, j].set(sparsity_type)
    structure = jnp.zeros(4, dtype=jnp.int32) # jnp.concatenate([_invar_shape, _outvar_shape]) 
    edges = edges.at[1:, i, j].set(structure)
    return edges

vertex_registry[lax.stop_gradient_p] = add_stop_gradient_vertex


def add_copy_gradient_vertex(edges, eqn, variables):
    """
    TODO check this for correctness
    Adds a vertex for operations that are essentially just copies the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    filtered_invars = filter_invars(eqn, variables)
    # Handle literal inputs
    if len(filtered_invars) == 0:
        return edges
    _invar_shape = get_shape(filtered_invars[0])
    _outvar_shape = get_shape(eqn.outvars[0])
    
    sparsity_type = 8
                    
    num_i = edges.at[0, 0, 0].get()
    i = variables[str(filtered_invars[0])]
    j = variables[str(eqn.outvars[0])] - num_i - 1

    edges = edges.at[0, i, j].set(sparsity_type)
    structure = jnp.concatenate([_invar_shape, _outvar_shape]) 
    edges = edges.at[1:, i, j].set(structure)
    return edges


vertex_registry[lax.broadcast_in_dim_p] = add_copy_gradient_vertex
vertex_registry[lax.squeeze_p] = add_copy_gradient_vertex
vertex_registry[lax.reshape_p] = add_copy_gradient_vertex

# Reshaping of tensors. Does not change the Jacobian accumulation as slicing also
# merely copies the respective partials. However, it terminates the derivative
# flow over "sliced-off" edges.   
vertex_registry[lax.slice_p] = add_copy_gradient_vertex
vertex_registry[lax.dynamic_slice_p] = add_copy_gradient_vertex
vertex_registry[lax.dynamic_update_slice_p] = add_copy_gradient_vertex
vertex_registry[lax.convert_element_type_p] = add_copy_gradient_vertex


# NOTE not sure about these guys
vertex_registry[lax.pad_p] = add_copy_gradient_vertex


def add_concatenate_vertex(edges, eqn, variables):
    """
    NOTE: Currently not working!
    Adds a vertex for operations that are essentially just copy the gradient 
    such as squeeze, broadcast_in_dim etc.
    """
    # Run loop over all values that are concatenated
    _outvar_shape = get_shape(eqn.outvars[0])
    filtered_invars = filter_invars(eqn, variables)
    for invar in filtered_invars:
        _invar_shape = get_shape(filtered_invars[0])

        sparsity_type = 0  # TODO this is the important bit!
        num_i = edges.at[0, 0, 0].get()
        i = variables[str(invar)]
        j = variables[str(eqn.outvars[0])] - num_i - 1

        edges = edges.at[0, i, j].set(sparsity_type)
        structure = jnp.concatenate([_invar_shape, _outvar_shape])
        edges = edges.at[1:, i, j].set(structure)
                        
    return edges

vertex_registry[lax.concatenate_p] = add_concatenate_vertex  


def add_zero_vertex(edges, eqn, variables):
    return edges

vertex_registry[lax.iota_p] = add_zero_vertex
    
