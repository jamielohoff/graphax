import jax
import jax.lax as lax
import jax.numpy as jnp

from chex import Array

from ..core import vertex_eliminate, get_shape


def scan(f, init, xs, length=None):
    if xs is None:
        xs = [None] * length
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.stack(ys)


def cond(condition, true_fn, false_fn, *xs):
    if condition:
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
    num_i, num_vo = get_shape(edges)
    
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
    vertices = jnp.arange(1, num_vo+1)
    edges, preelim_order = lax.scan(loop_fn, edges, vertices)
    if return_preeliminated:
        return edges, [int(p) for p in preelim_order if p > 0]
    return edges


def update_edges(vertex: int, edges: Array):
    dead_branch = is_dead_branch(vertex, edges)
    edges, vert = lax.cond(dead_branch,
                            lambda v, e: (vertex_eliminate(v, e)[0], v), 
                            lambda v, e: has_markowitz_degree_1(v, e), 
                            vertex, edges)        
    return edges, vert
    

def is_dead_branch(vertex: int, edges: Array) -> bool:
    # Remove dead branches from the computational graph
    num_i, num_vo = get_shape(edges)
    row_flag = jnp.sum(edges[0, vertex+num_i, :]) > 0
    col_flag = jnp.sum(edges[0, 1:, vertex-1]) == 0
    return jnp.logical_and(row_flag, col_flag)


def has_markowitz_degree_1(vertex: int, edges: Array):
    markowitz_degree_1 = check_markowitz_degree(vertex, edges)
    edges, vert = lax.cond(markowitz_degree_1,
                            lambda v, e: check_sparsity(v, e), 
                            lambda v, e: (e, -1), 
                            vertex, edges)        
    return edges, vert


def check_sparsity(vertex: int, edges: Array):
    sparse = is_sparse(vertex, edges)
    edges, vert = lax.cond(sparse,
                            lambda v, e: (vertex_eliminate(v, e)[0], v), 
                            lambda v, e: (e, -1), 
                            vertex, edges)        
    return edges, vert


def check_markowitz_degree(vertex: int, edges: Array) -> bool:
    # Check if vertex has only one input and one output
    # In a sense this is similar to Markowitz degree 1
    num_i, num_vo = get_shape(edges)
    row = edges[0, vertex+num_i, :]
    col = edges[0, 1:, vertex-1]
       
    row_flag = jnp.equal(jnp.sum(jnp.where(row > 0, 1, 0)), 1)
    col_flag = jnp.equal(jnp.sum(jnp.where(col > 0, 1, 0)), 1)
    return jnp.logical_and(col_flag, row_flag)


SPARSITY_MAP = jnp.array([[2, 4], # sparsity type = 2
                          [1, 3], # sparsity type = 3
                          [2, 3], # sparsity type = 4
                          [1, 4]]) # sparsity type = 5


def is_sparse(vertex: int, edges: Array):
    num_i, num_vo = get_shape(edges)
    in_edge_idx = jnp.nonzero(edges.at[0, vertex+num_i, :].get(), size=1, fill_value=0)[0][0]
    out_edge_idx = jnp.nonzero(edges.at[0, 1:, vertex-1].get(), size=1, fill_value=0)[0][0]
            
    in_edge = edges.at[:, vertex+num_i, in_edge_idx].get()
    out_edge = edges.at[:, out_edge_idx+1, vertex-1].get()
                
    in_edge_flag = _sparsity_checker(in_edge)    
    out_edge_flag = _sparsity_checker(out_edge) 
    
    return jnp.logical_and(in_edge_flag, out_edge_flag)


# TODO make this recursive!
def _sparsity_checker(edge: Array):
    sparsity_type = edge.at[0].get()
    return lax.cond(jnp.logical_or(sparsity_type == 6, sparsity_type == 7),
                    lambda e: True,
                    lambda e: _sparsity(e),
                    edge)

def _sparsity(edge: Array):
    sparsity_type = edge.at[0].get()
    return lax.cond(sparsity_type > 1,
                    lambda e: _parallel_sparsity(e),
                    lambda e: _single_sparsity(e),
                    edge)

def _parallel_sparsity(edge: Array):
    sparsity_type = edge.at[0].get()
    idxs = SPARSITY_MAP[sparsity_type-2]
    _in_shape = edge.at[idxs[0]].get()
    _out_shape = edge.at[idxs[1]].get()
    return lax.cond(jnp.logical_and(_in_shape == 1, _out_shape == 1),
                    lambda e: True,
                    lambda e: _single_sparsity(e),
                    edge)
    

def _single_sparsity(edge: Array):
    return lax.cond(jnp.prod(edge) == 1, lambda: True, lambda: False)
    
