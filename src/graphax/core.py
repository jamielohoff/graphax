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

from jax._src import linear_util as lu
from jax._src.util import unzip2
from jax._src.api_util import argnums_partial
import jax._src.interpreters.partial_eval as pe


# elemental_rules = {}

def tree_allclose(tree1, tree2, equal_nan: bool = False) -> bool:
    allclose = lambda a, b: jnp.allclose(a, b, equal_nan=equal_nan, atol=1e-5, rtol=1e-4)
    is_equal = jtu.tree_map(allclose, tree1, tree2)
    return jtu.tree_reduce(jnp.logical_and, is_equal)


class CCETracer(core.Tracer):
    __slots__ = ["primal", "elemental"]
    def __init__(self, trace, primal, elemental):
        self._trace = trace
        self.primal = primal
        print("primal at construction:", primal)
        self.elemental = elemental
        
    @property
    def aval(self):
        return core.get_aval(self.primal)
    
    def full_lower(self):
        return core.full_lower(self.primal)
    

class CCETrace(core.Trace):

    def pure(self, val):
        return CCETracer(self, val, 0)
    
    def lift(self, val):
        return CCETracer(self, val, 0)
    
    def sublift(self, val):
        return CCETracer(self, val.primal, val.elemental)
    
    def process_primitive(self, primitive, tracers, params):
        elemental_rule = elemental_rules.get(primitive)
        primal_out, elemental_out = elemental_rule([t.primal for t in tracers], **params)
        # primal_out = primitive.bind(*[t.primal for t in tracers], **params)
        print("primitive", primitive)
        print("process primitive primal_out:", primal_out)
        print("process primitive elemental_out:", elemental_out)
        # if primitive.multiple_results:
        # print("CCE", [CCETracer(self, p, e) for p, e in zip(primal_out, elemental_out)])
        # return [CCETracer(self, p, e) for p, e in zip(primal_out, elemental_out)]
        # else:
        print("CCE", CCETracer(self, primal_out[0], elemental_out[0]))
        return primal_out # CCETracer(self, primal_out[0], elemental_out[0])


def _jacve(fun: Callable) -> Callable:
    @wraps(fun)
    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun)

        f_jac = _cce(f, *args)
        print("Result:", f_jac(*args)) # *args
        return f_jac(*args) # *args
    
    return jacfun


from jax.tree_util import tree_flatten
from jax._src.api_util import flatten_fun_nokwargs
from jax._src import dispatch


def _cce(fun: lu.WrappedFun, *primals):
    primals_flat, in_tree = tree_flatten(primals)
    for arg in primals_flat: dispatch.check_arg(arg)
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    jac_fn = cce(flat_fun, primals_flat)
    out_tree = out_tree()
    # out_primals, jac, aux = cce(fun, primals, has_aux=has_aux)
    return jac_fn


###
def cce(traceable, primals,):
    pvals, jaxpr, consts = ccetrace(traceable, *primals)
    return partial(core.eval_jaxpr, jaxpr, consts)


from jax._src.api_util import flatten_fun
from jax.tree_util import tree_unflatten


# aka linearize
def ccetrace(traceable, *primals, **kwargs):
    _traceable = cce_subtrace(traceable)
    print("traceable:", _traceable)
    cce_fun = ccefun(_traceable)
    print("cce_fun:", cce_fun)

    # We have to use pe.PartialVal.unknown to enable jaxpr tracing
    in_pvals = tuple(pe.PartialVal.unknown(core.get_aval(p)) for p in primals)
    print("in_pvals:", in_pvals)

    # Inputs have to have the shape *args, **kwargs with {} representing no kwargs
    _, in_tree = tree_flatten((primals, {}))
    ccefun_flat, out_tree = flatten_fun(cce_fun, in_tree)

    jaxpr, out_pvals, consts = pe.trace_to_jaxpr_nounits(ccefun_flat, in_pvals)
    print("out_pvals:", out_pvals)
    print("consts:", consts)
    print(jaxpr)

    out_primals_pvals = tree_unflatten(out_tree(), out_pvals)
    out_primals_consts = [pval.get_known() for pval in out_primals_pvals]
    print("out_primals_pvals:", out_primals_pvals)
    print("out_primals_consts:", out_primals_consts)
    return out_primals_pvals, jaxpr, consts

from jax._src import source_info_util
import contextlib

@lu.transformation
def ccefun(*primals):
    ctx = contextlib.nullcontext()
    with core.new_main(CCETrace) as main, ctx:
        out_primals = yield (main, primals), {}
        del main
    # out_tangents = [instantiate_zeros(t) if inst else t for t, inst
    #               in zip(out_tangents, instantiate)]
    # print("ccefun out_primals:", out_primals)
    yield out_primals


@lu.transformation
def cce_subtrace(main, primals):
    trace = CCETrace(main, core.cur_sublevel())
    for x in list(primals):
        if isinstance(x, core.Tracer):
            if x._trace.level >= trace.level:
                raise core.escaped_tracer_error(
                    x, f"Tracer from a higher level: {x} in trace {trace}")
            assert x._trace.level < trace.level

    in_tracers = [CCETracer(trace, x, 0)
                for x in primals]
    ans = yield in_tracers, {}
    out_tracers = map(trace.full_raise, ans)
    # print("out_tracers:", list(out_tracers))
    yield [t.primal for t in out_tracers]


def jacve(fun: Callable, 
            order: Union[Sequence[int], str], 
            argnums: Sequence[int] = (0,),
            count_ops: bool = False,
            dense_representation: bool = True) -> Callable:
    @wraps(fun)
    def jacfun(*args, **kwargs):

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
    return jacfun


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
    _checkify_tensor(new_post)
    return new_post


def unload_pre_transforms(post, pre, iota):
    new_pre = pre.copy()
    for transform in post.pre_transforms:
        new_pre = transform.apply(new_pre, iota)
    _checkify_tensor(new_pre)
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
    everything that has a _val in its name is a SparseTensor object

    Args:
        vertex (_type_): _description_
        jaxpr (_type_): _description_
        graph (_type_): _description_
        transpose_graph (_type_): _description_
        iota (_type_): _description_
        vo_vertices (_type_): _description_
    """
    # print("vertex:", vertex)
    eqn = jaxpr.eqns[vertex-1]
    # print(eqn.primitive)
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

            # print(in_edge.count, "->", eqn.outvars[0].count, "->", out_edge.count)
            # print("Post:", _post_val)
            # print("Pre:", _pre_val) 
            
            if len(pre_val.post_transforms) > 0 and post_val.val is not None:
                _post_val = unload_post_transforms(post_val, pre_val, iota)
                
            if len(post_val.pre_transforms) > 0 and pre_val.val is not None:
                _pre_val = unload_pre_transforms(post_val, pre_val, iota)
                                
            # Multiply the two values of the edges if applicable
            if pre_val.val is not None and post_val.val is not None:     
                edge_outval = _post_val * _pre_val
                num_mul += get_num_muls(_post_val, _pre_val)
                    
            elif pre_val.val is not None:
                edge_outval = _pre_val  
            else:
                edge_outval = _post_val
                
            # print("Edge_outval:", edge_outval)
            # Offload the remain Jacobian transforms to the output tensor
            if len(post_val.post_transforms) > 0:
                edge_outval = prepend_post_transforms(post_val, edge_outval, iota)

            if len(pre_val.pre_transforms) > 0:
                edge_outval = append_pre_transforms(pre_val, edge_outval, iota)
                                                
            # If there is already an edge between the two vertices, add the new
            # edge to the existing one
            if graph.get(in_edge).get(out_edge) is not None:
                _edge = transpose_graph[out_edge][in_edge]  
                # print("Edge_outval:", edge_outval)      
                # print("Edge:", _edge)  
  
                # Offload the remaining Jacobian transforms to the output tensor
                if len(edge_outval.post_transforms) > 0:
                    for transform in edge_outval.post_transforms:
                        edge_outval = transform.apply(edge_outval, iota)

                if len(edge_outval.pre_transforms) > 0:
                    for transform in edge_outval.pre_transforms[::-1]: # Do we need the [::-1] here?
                        edge_outval = transform.apply_inverse(edge_outval, iota)
                
                # Offload the remain Jacobian transforms to the output tensor
                if len(_edge.post_transforms) > 0:
                    for transform in _edge.post_transforms:
                        _edge = transform.apply(_edge, iota)

                if len(_edge.pre_transforms) > 0:
                    for transform in _edge.pre_transforms:
                        _edge = transform.apply_inverse(_edge, iota)

                _checkify_tensor(edge_outval)
                edge_outval += _edge
                num_add += get_num_adds(edge_outval, _edge)
                
            _checkify_tensor(edge_outval)
            # print("Edge_outval:", edge_outval)
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


def _build_graph(env, graph, transpose_graph, jaxpr, args, consts, var_id, vo_vertices, counter, level=0):
    """This function performs the tracing of the jaxpression and it's transformation
    into a computational graph.

    env stores primal values
    graph, graph_transpose store partial derivatives
    """
    
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
        _checkify_tensor(val)
        if isinstance(invar, core.Var):
            graph[invar][outvar] = val
            transpose_graph[outvar][invar] = val
                            
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    # NOTE: this is essentially the tracing part. Probably should write a proper
    # tracing system with lift etc. for better compatibility with JAX
    # Loop though elemental partials and create an abstract representation of
    # the computational graph
    for eqn in jaxpr.eqns:
        # Treatment of intermediate variables that are also output variables
        for outvar in eqn.outvars:
            if type(outvar) is core.Var and outvar not in var_id.keys():
                var_id[outvar] = counter
                counter += 1
                    
        for invar in eqn.invars:
            if invar in jaxpr._outvars:
                vertex = var_id[invar]
                vo_vertices.add(vertex)
                
        print("eqn:", eqn)
        # print("invars", eqn.invars)
        # print("outvars", eqn.outvars)
        invals = safe_map(read, eqn.invars)              
        if eqn.primitive is lax.stop_gradient_p:
            primal_outvals = lax.stop_gradient_p.bind(*invals, **eqn.params)
            safe_map(write, eqn.outvars, [primal_outvals])
            
        elif type(eqn.primitive) is core.AxisPrimitive:
            print("invals", invals)
            primal_outvals = jax._src.pjit.pjit_p.bind(*invals, **eqn.params)
            safe_map(write, eqn.outvars, primal_outvals)
            subjaxpr = eqn.params["jaxpr"].jaxpr
            # primal_outvals, elemental_outvals = elemental_rules[eqn.primitive](invals, **eqn.params)
            _build_graph(env, graph, transpose_graph, subjaxpr, invals, consts, 
                        var_id, vo_vertices, counter, level+1)
            # print("primal_outvals:", primal_outvals)
            # print("elemental_outvals:", elemental_outvals)

            #  invars = [invar for invar in eqn.invars if type(invar) is core.Var]
            # NOTE: Currently only able to treat one output variable

            # _write_elemental = partial(write_elemental, eqn.outvars[0])
            # if len(invars) == len(elemental_outvals):
            #     safe_map(_write_elemental, invars, elemental_outvals)
            
        else:
            if eqn.primitive not in elemental_rules:
                raise NotImplementedError(f"{eqn.primitive} does not have registered elemental partial.")
            print("invals:", invals)
            primal_outvals, elemental_outvals = elemental_rules[eqn.primitive](invals, **eqn.params)
            safe_map(write, eqn.outvars, [primal_outvals])
            invars = [invar for invar in eqn.invars if type(invar) is core.Var]
            # NOTE: Currently only able to treat one output variable

            _write_elemental = partial(write_elemental, eqn.outvars[0])
            if len(invars) == len(elemental_outvals):
                safe_map(_write_elemental, invars, elemental_outvals)


def _prune_graph(graph, transpose_graph, jaxpr, argnums):
    # TODO implement proper pruning that does not accidentially kill off edges                  
    # Prune the computational graph
    has_dead_vertices = True
    for i, invar in enumerate(jaxpr.invars):
        if i not in argnums:
            for in_edge in transpose_graph[invar].keys():
                del graph[in_edge][invar]
            for out_edge in graph[invar].keys():   
                del transpose_graph[out_edge][invar]   
                
            del graph[invar]
            del transpose_graph[invar]
            print("Pruned input variable:", invar)
        
    already_deleted = []
    while has_dead_vertices:
        to_delete = []
        for eqn in jaxpr.eqns:
            ov = eqn.outvars[0]
            if ov not in jaxpr.outvars and ov not in already_deleted:
                if len(graph[ov]) == 0 or len(transpose_graph[ov]) == 0:
                    to_delete.append(ov) 
                    
        if len(to_delete) > 0:
            for ov in to_delete:
                for in_edge in transpose_graph[ov].keys():
                    del graph[in_edge][ov]
                for out_edge in graph[ov].keys():   
                    del transpose_graph[out_edge][ov]   
                    
                del graph[ov]
                del transpose_graph[ov] 
                print("Pruned output variable:", ov)
            already_deleted.extend(to_delete)
        else:
            has_dead_vertices = False


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
        
    vo_vertices = set() # contains all intermediate and output vertices
    counter = 1
    var_id = {}
    
    jaxpr_invars = [invar for i, invar in enumerate(jaxpr.invars) if i in argnums]
    _build_graph(env, graph, transpose_graph, jaxpr, args, consts, var_id, vo_vertices, counter)
    _prune_graph(graph, transpose_graph, jaxpr, argnums)
    
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
                        for transform in tensor.pre_transforms[::-1]: # Do we need the [::-1] here?
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
        jac_vals = [graph[invar][outvar]
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

