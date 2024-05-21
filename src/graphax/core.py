from typing import Callable, Sequence, Union
from functools import wraps, partial
from collections import defaultdict

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu

from jax._src.util import safe_map
import jax._src.core as core

from .primitives import elemental_rules
from .sparse.tensor import get_num_muls, get_num_adds, _checkify_tensor
from .sparse.utils import zeros_like, get_largest_tensor


def tree_allclose(tree1, tree2, equal_nan: bool = False):
    allclose = lambda a, b: jnp.allclose(a, b, equal_nan=equal_nan, atol=1e-5, rtol=1e-4)
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


def jacve(fun: Callable, 
            order: Union[Sequence[int], str], 
            argnums: Sequence[int] = (0,),
            count_ops: bool = False,
            dense_representation: bool = True) -> Callable:
    @wraps(fun)
    def wrapped(*args, **kwargs):
        # TODO Make repackaging work properly with one input value only
        flattened_args, in_tree = jtu.tree_flatten(args)
        closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)

        out = vertex_elimination_jaxpr(closed_jaxpr.jaxpr, 
                                        order, 
                                        closed_jaxpr.literals, 
                                        *args, 
                                        argnums=argnums,
                                        count_ops=count_ops,
                                        dense_representation=dense_representation)
        if count_ops: 
            out, op_counts = out
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if len(closed_jaxpr.jaxpr.outvars) == 1:
                return out[0], op_counts
            return jtu.tree_unflatten(out_tree, out), op_counts
        else:
            out_tree = jtu.tree_structure(tuple(closed_jaxpr.jaxpr.outvars))
            if len(closed_jaxpr.jaxpr.outvars) == 1:
                return out[0]
            return jtu.tree_unflatten(out_tree, out)
    return wrapped


def _iota_shape(jaxpr, argnums):
    largest_input = get_largest_tensor([jaxpr._invars[arg] for arg in argnums])
    largest_output = get_largest_tensor(jaxpr._outvars)
        
    # TODO check if this is meaningful
    if largest_input == 1 and largest_output == 1:
        return None
    elif largest_output == 1:
        return jnp.ones((1, largest_input))
    elif largest_input == 1:
        return jnp.ones((largest_output, 1))
    else:
        return jnp.eye(max(largest_output, largest_input), largest_input) 
        
        
def unload_post_transforms(post, pre, iota):
    new_post = post.copy()
    for transform in pre.post_transforms:
        new_post = transform.apply_inverse(new_post, iota)
    return new_post


def unload_pre_transforms(post, pre, iota):
    new_pre = pre.copy()
    for transform in post.pre_transforms:
        new_pre = transform.apply(new_pre, iota)
    return new_pre


def prepend_post_transforms(post, out, iota):
    transforms = post.post_transforms + out.post_transforms
    out.post_transforms = transforms
    return out


def append_pre_transforms(pre, out, iota):
    transforms = pre.pre_transforms + out.pre_transforms
    out.pre_transforms = transforms
    return out
        
    
def _eliminate_vertex(vertex, jaxpr, graph, transpose_graph, iota, vo_vertices):
    """Function that eliminates a vertex from the computational graph.

    Args:
        vertex (_type_): _description_
        jaxpr (_type_): _description_
        graph (_type_): _description_
        transpose_graph (_type_): _description_
        iota (_type_): _description_
        vo_vertices (_type_): _description_
        counters (_type_): _description_
    """
    eqn = jaxpr.eqns[vertex-1]
    num_mul, num_add = 0, 0
    for out_edge in graph[eqn.outvars[0]].keys():
        post_val = graph[eqn.outvars[0]][out_edge].copy()
        for in_edge in transpose_graph[eqn.outvars[0]].keys():
            pre_val = transpose_graph[eqn.outvars[0]][in_edge].copy()
            
            # TODO implement a process that discards unnecessary edges from the computation
            
            # Handle stuff like reshape, squeeze etc.            
            # Apply Jacobian transforms where applicable
            _pre_val = pre_val.copy()
            _post_val = post_val.copy()
            
            print("Post:", _post_val)
            print("Pre:", _pre_val) 
            
            if len(pre_val.post_transforms) > 0 and post_val.val is not None:
                _post_val = unload_post_transforms(post_val, pre_val, iota)
                
            if len(post_val.pre_transforms) > 0 and pre_val.val is not None:
                _pre_val = unload_pre_transforms(post_val, pre_val, iota)
                                
            # Multiply the two values of the edges if applicable
            if pre_val.val is not None and post_val.val is not None:     
                edge_outval = _post_val * _pre_val
                # print("num_muls:", get_num_muls(_post_val, _pre_val))
                num_mul += get_num_muls(_post_val, _pre_val)
                    
            elif pre_val.val is not None:
                edge_outval = _pre_val  
            else:
                edge_outval = _post_val
                
            # Offload the remain Jacobian transforms to the output tensor
            if len(post_val.post_transforms) > 0:
                edge_outval = prepend_post_transforms(post_val, edge_outval, iota)

            if len(pre_val.pre_transforms) > 0:
                edge_outval = append_pre_transforms(pre_val, edge_outval, iota)
                                                
            # If there is already an edge between the two vertices, add the new
            # edge to the existing one
            if graph.get(in_edge).get(out_edge) is not None:
                _edge = transpose_graph[out_edge][in_edge]                
                # Offload the remain Jacobian transforms to the output tensor
                if len(edge_outval.post_transforms) > 0:
                    for transform in edge_outval.post_transforms:
                        edge_outval = transform.apply(edge_outval, iota)

                if len(edge_outval.pre_transforms) > 0:
                    for transform in edge_outval.pre_transforms:
                        edge_outval = transform.apply_inverse(edge_outval, iota)
                
                # Offload the remain Jacobian transforms to the output tensor
                if len(_edge.post_transforms) > 0:
                    for transform in _edge.post_transforms:
                        _edge = transform.apply(_edge, iota)

                if len(_edge.pre_transforms) > 0:
                    for transform in _edge.pre_transforms:
                        _edge = transform.apply_inverse(_edge, iota)

                _checkify_tensor(edge_outval)
                print("Edge_outval:", edge_outval)
                print("Edge:", _edge)
                edge_outval += _edge
                num_add += get_num_adds(edge_outval, _edge)
                
            _checkify_tensor(edge_outval)
            graph[in_edge][out_edge] = edge_outval
            transpose_graph[out_edge][in_edge] = edge_outval
                
    # Cleanup of input and output edges
    if vertex not in vo_vertices:
        for in_vertex in transpose_graph[eqn.outvars[0]].keys():
            del graph[in_vertex][eqn.outvars[0]]
    for out_vertex in graph[eqn.outvars[0]].keys():    
        del transpose_graph[out_vertex][eqn.outvars[0]]
    
    # Cleanup the eliminated vertex
    del graph[eqn.outvars[0]]
    if vertex not in vo_vertices:
        del transpose_graph[eqn.outvars[0]]
    # print(vertex, ":", num_mul)
    return num_mul, num_add


def _checkify_order(order, jaxpr, vo_vertices):
    if type(order) is str:
        if order == "forward" or order == "fwd":
            return [i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices]
        elif order == "reverse" or order == "rev":
            return [i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices][::-1]
        else:
            raise ValueError(f"{order} is not a valid order identifier!")
    else:
        vertex_set = set([i for i, eqn in enumerate(jaxpr.eqns, start=1) 
                    if eqn.outvars[0] not in jaxpr.outvars or i in vo_vertices])
        set_from_order = set(order)
        missing_vertices = vertex_set.difference(set_from_order)
        if len(missing_vertices) > 0:
            raise ValueError(f"Supplied order is missing vertices {missing_vertices}!")
    return order


def vertex_elimination_jaxpr(jaxpr: core.Jaxpr, 
                            order: Union[Sequence[int], str], 
                            consts: Sequence[core.Literal], 
                            *args, 
                            argnums: Sequence[int] = (0,),
                            count_ops: bool = False,
                            dense_representation: bool = True):    
    env = {}
    graph = defaultdict(lambda: defaultdict()) # Input connectivity
    transpose_graph = defaultdict(lambda: defaultdict()) # Output connectivity  
        
    # Reads variable and corresponding traced shaped array
    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    # Adds new variable and corresponding traced shaped array
    def write(var, val):
        env[var] = val
        
    # Writes a new elemental partial to the graph and transpose_graph
    def write_elemental(outvar, invar, val):
        if isinstance(invar, core.Var):
            graph[invar][outvar] = val
            transpose_graph[outvar][invar] = val
                            
    jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    vo_vertices = set() # contains all intermediate and output vertices
    counter = 1
    var_id = {}
    
    # NOTE: this is essentially the tracing part. Probably should write a proper
    # tracing system with lift etc. for better compatibility with JAX
    # Loop though elemental partials and create an abstract representation of
    # the computational graph
    for i, eqn in enumerate(jaxpr.eqns):
        # Treatment of intermediate variables that are also output variables
        for outvar in eqn.outvars:
            if type(outvar) is core.Var and outvar not in var_id.keys():
                var_id[outvar] = counter
                counter += 1
                    
        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)
                
        invals = safe_map(read, eqn.invars)              
        
        if eqn.primitive is lax.stop_gradient_p:
            primal_outvals = lax.stop_gradient_p.bind(*invals, **eqn.params)
            safe_map(write, eqn.outvars, [primal_outvals])
        else:
            if eqn.primitive not in elemental_rules:
                raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")
            
            primal_outvals, elemental_outvals = elemental_rules[eqn.primitive](invals, **eqn.params)
            safe_map(write, eqn.outvars, [primal_outvals])
            invars = [invar for invar in eqn.invars if type(invar) is core.Var]
            # NOTE: Currently only able to treat one output variable
            
            _write_elemental = partial(write_elemental, eqn.outvars[0])
            if len(invars) == len(elemental_outvals):
                safe_map(_write_elemental, invars, elemental_outvals)
      
    # TODO implement proper pruning that does not accidentially kill off edges                  
    # # Prune the computational graph
    # has_dead_vertices = True
    # for i, invar in enumerate(jaxpr.invars):
    #     if i not in argnums:
    #         for in_edge in transpose_graph[invar].keys():
    #             del graph[in_edge][invar]
    #         for out_edge in graph[invar].keys():   
    #             del transpose_graph[out_edge][invar]   
                
    #         del graph[invar]
    #         del transpose_graph[invar]
        
    # already_deleted = []
    # while has_dead_vertices:
    #     to_delete = []
    #     for eqn in jaxpr.eqns:
    #         o = eqn.outvars[0]
    #         if o not in jaxpr.outvars and o not in already_deleted:
    #             if len(graph[o]) == 0 or len(transpose_graph[o]) == 0:
    #                 to_delete.append(o) 
                    
    #     if len(to_delete) > 0:
    #         for o in to_delete:
    #             for in_edge in transpose_graph[o].keys():
    #                 del graph[in_edge][o]
    #             for out_edge in graph[o].keys():   
    #                 del transpose_graph[out_edge][o]   
                    
    #             del graph[o]
    #             del transpose_graph[o] 
    #         already_deleted.extend(to_delete)
    #     else:
    #         has_dead_vertices = False
    
    iota = _iota_shape(jaxpr, argnums)
        
    # Eliminate the vertices
    num_muls, num_adds = 0, 0
    counts = []
    order = _checkify_order(order, jaxpr, vo_vertices)
    for vertex in order:
        num_mul, num_add = _eliminate_vertex(vertex, jaxpr, graph, transpose_graph, iota, vo_vertices)
        if count_ops:
            counts.append((num_mul, num_add))
            num_muls += num_mul
            num_adds += num_add
           
    # Offloading all remaining Jacobian transforms to the output variables 
    # before densification!    
    for invar in jaxpr_invars:
        for outvar in jaxpr.outvars:
            if graph.get(invar) is not None:
                if graph.get(invar).get(outvar) is not None:
                    tensor = graph[invar][outvar].copy()
                    if len(tensor.pre_transforms) > 0:
                        for transform in tensor.pre_transforms:
                            tensor = transform.apply_inverse(tensor, iota)
                    if len(tensor.post_transforms) > 0:
                        for transform in tensor.post_transforms:
                            tensor = transform.apply(tensor, iota)
                    graph[invar][outvar] = tensor
    
    # Collect outputs  
    if dense_representation:   
        jac_vals = [graph[invar][outvar].dense(iota) 
                    if outvar in list(graph[invar].keys()) else zeros_like(outvar, invar)
                    for outvar in jaxpr.outvars for invar in jaxpr_invars]
    else:
        jac_vals = [graph[invar][outvar].val
                    if outvar in list(graph[invar].keys()) else None
                    for outvar in jaxpr.outvars for invar in jaxpr_invars]
        
    # Restructure Jacobians for more complicated pytrees
    n = len(jaxpr_invars)
    if n > 1:
        ratio = len(jac_vals)//n
        jac_vals = [tuple(jac_vals[i*n:i*n+n]) for i in range(0, ratio)]
        
    if count_ops:
        order_counts = [(int(o), int(c[0])) for o, c in zip(order, counts)]
        aux = {"num_muls": num_muls, 
                "num_adds": num_adds, 
                "order_counts": order_counts}
        return jac_vals, aux

    return jac_vals

