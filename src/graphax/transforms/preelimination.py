import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from ..core import vertex_eliminate


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def cond(cond, true_fn, false_fn, *xs):
    if cond:
        return true_fn(*xs)
    else:
        return false_fn(*xs)


def safe_preeliminations(edges: Array, return_preeliminated: bool = False) -> Array:
    """
    Function that runs a safe-preelimination routine that eliminates all vertices
    with only Kronecker symbols or only one scalar input and one scalar output.
    
    Thus we perform safe preeliminations on the all vertices that have only 
    one ingoing and one outgoing edge if:
    1.) The Jacobian is a scalar, i.e. has shape (1, 1, 1, 1, 1)
    2.) Is of sparsity type 6, 7 or 8 (Parallel matrix operations)
    3.) Is of sparsity type 2, 3, 4, 5 with the dense components having size 1 
    (Parallel vector operations)
    """
        
    def loop_fn(carry, vertex):
        _edges = carry
        # Do not preeliminate output vertices
        is_output_vertex = _edges.at[2, 0, vertex-1].get() > 0
        _edges, _vert = lax.cond(is_output_vertex,
                                lambda v, e: (e, -1), 
                                lambda v, e: update_edges(v, e), 
                                vertex, _edges)        
        
        carry = _edges
        return carry, _vert
    vertices = jnp.arange(1, edges.at[0, 0, 1].get()+1)
    edges, preelim_order = lax.scan(loop_fn, edges, vertices)
    if return_preeliminated:
        return edges, preelim_order
    return edges


def update_edges(vertex: int, edges: Array):
    dead_branch = is_dead_branch(vertex, edges)
    edges, vert = lax.cond(dead_branch,
                            lambda v, e: (vertex_eliminate(v, e)[0], v), 
                            lambda v, e: is_eligible(v, e), 
                            vertex, edges)        
    return edges, vert
    

def is_dead_branch(vertex: int, edges: Array) -> bool:
    # Remove dead branches from the computational graph
    num_i = edges.at[0, 0, 0].get()
    row_flag = jnp.sum(edges[0, vertex+num_i, :]) > 0
    col_flag = jnp.sum(edges[0, 1:, vertex-1]) == 0
    return jnp.logical_and(row_flag, col_flag)


def is_eligible(vertex: int, edges: Array):
    markowitz_degree_1 = check_markowitz(vertex, edges)
    edges, vert = lax.cond(markowitz_degree_1,
                            lambda v, e: eliminate(v, e), 
                            lambda v, e: (e, -1), 
                            vertex, edges)        
    return edges, vert


def eliminate(vertex: int, edges: Array):
    is_sparse = check_sparsity(vertex, edges)
    edges, vert = lax.cond(is_sparse,
                            lambda v, e: (vertex_eliminate(v, e)[0], v), 
                            lambda v, e: (e, -1), 
                            vertex, edges)        
    return edges, vert


def check_markowitz(vertex: int, edges: Array) -> bool:
    # Check if vertex has only one input and one output
    # In a sense this is similar to Markowitz degree 1
    num_i = edges.at[0, 0, 0].get()
    row = edges[0, vertex+num_i, :]
    col = edges[0, 1:, vertex-1]
       
    row_flag = jnp.equal(jnp.sum(jnp.where(row > 0, 1, 0)), 1)
    col_flag = jnp.equal(jnp.sum(jnp.where(col > 0, 1, 0)), 1)
    return jnp.logical_and(col_flag, row_flag)


SPARSITY_MAP = jnp.array([[2, 4], # sparsity type = 2
                          [1, 3], # sparsity type = 3
                          [2, 3], # sparsity type = 4
                          [1, 4]]) # sparsity type = 5


def check_sparsity(vertex: int, edges: Array):
    num_i = edges.at[0, 0, 0].get()
    in_edge_idx = jnp.nonzero(edges.at[0, vertex+num_i, :].get(), size=1, fill_value=0)[0][0]
    out_edge_idx = jnp.nonzero(edges.at[0, :, vertex-1].get(), size=1, fill_value=0)[0][0]
        
    in_edge = edges.at[:, vertex+num_i, in_edge_idx].get()
    out_edge = edges.at[:, out_edge_idx, vertex-1].get()
            
    in_edge_flag = _sparsity_checker(in_edge)    
    out_edge_flag = _sparsity_checker(out_edge) 
    
    return jnp.logical_and(in_edge_flag, out_edge_flag)


# TODO make this recursive!
def _sparsity_checker(edge: Array):
    sparsity_type = edge.at[0].get()
    is_parallel = lax.cond(jnp.logical_or(sparsity_type == 6, sparsity_type == 7),
                            lambda: True,
                            lambda: False)
    
    idxs = SPARSITY_MAP[sparsity_type-2]
    _in = edge.at[idxs[0]].get()
    _out = edge.at[idxs[1]].get()
    is_parallel = lax.cond(jnp.logical_and(_in == 1, _out == 1),
                            lambda x: True,
                            lambda x: x,
                            is_parallel)
    
    is_parallel = lax.cond(jnp.prod(edge) == 1,
                            lambda x: True,
                            lambda x: x,
                            is_parallel)    
    
    return is_parallel
    
