from typing import Callable, Sequence

import jax
import jax.lax as lax
import jax.numpy as jnp

from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe

import jax._src.core as core
from jax._src.util import safe_map
from jax._src import custom_api_util
import jax._src.linear_util as lu
from jax._src.custom_derivatives import process_env_traces

map = safe_map
# Create new primitive for custom_elemental_call

# ------------------ custom_elemental implementation ------------------
custom_elemental_p = core.Primitive("custom_elemental")
custom_elemental_p.multiple_results = False

def _custom_elemental_abstract_eval(x, **params):
    print("x: ", x)
    return core.ShapedArray(x.shape, x.dtype)

custom_elemental_p.def_abstract_eval(_custom_elemental_abstract_eval)

def _custom_elemental_call_impl(*args):
    return jnp.sin(*args)

custom_elemental_p.def_impl(_custom_elemental_call_impl)


# register with base interpreter, jaxpr interpreter and batch interpreter
# write a custom_elemental decorator that wraps the function and the elemental



def dynamic_jaxpr_custom_elemental_call(primitive, fn: lu.WrappedFun, elemental, 
                                        tracers, *, symbolic_zeros):
    del elemental, symbolic_zeros
    return fn.call_wrapped(tracers)


class CustomElementalCallPrimitive(core.Primitive):
    multiple_results = True

    def bind(self, fn, elemental, *args, symbolic_zeros):
        args = map(core.full_lower, args)
        args = list(args)
        top_trace = core.find_top_trace(args)
        fn, env_trace_todo1 = process_env_traces(
            fn, self, top_trace and top_trace.level, False)
        elemental, env_trace_todo2 = process_env_traces(
            elemental, self, top_trace and top_trace.level, True)
        tracers = map(top_trace.full_raise, args)
        tracers = list(tracers)

        if isinstance(top_trace, jax)._src.interpreters.partial_eval.DynamicJaxprTrace:
            outs = dynamic_jaxpr_custom_elemental_call(self, fn, elemental, tracers,
                                                        symbolic_zeros=symbolic_zeros)
        _, env_trace_todo = lu.merge_linear_aux(env_trace_todo1, env_trace_todo2)
        print("outs: ", outs)
        return core.apply_todos(env_trace_todo, map(core.full_lower, outs))

    def impl(self, fn, _, *args):
        with core.new_sublevel():
            return fn.call_wrapped(*args)

    # def post_process(self, trace, out_tracers, jvp_was_run: bool):
    #     return trace.post_process_custom_jvp_call(out_tracers, jvp_was_run)

    def get_bind_params(self, params):
        new_params = dict(params)
        call_jaxpr = new_params.pop("call_jaxpr")
        num_consts = new_params.pop("num_consts")
        elemental_jaxpr_thunk = new_params.pop("elemental_jaxpr_thunk")
        fn = lu.wrap_init(core.jaxpr_as_fun(call_jaxpr))
        elemental = lambda x: x # lift_jvp(num_consts, jvp_jaxpr_thunk)
        return [fn, elemental], new_params


custom_elemental_call_p = CustomElementalCallPrimitive("custom_elemental_call")

def _custom_elemental_call_mlir_translation(ctx, *args, call_jaxpr, elemental_jaxpr_thunk,
                                            num_consts, symbolic_zeros):
    del elemental_jaxpr_thunk, num_consts, symbolic_zeros
    consts = mlir._ir_consts(call_jaxpr.consts)
    out, tokens = mlir.jaxpr_subcomp(ctx.module_context, call_jaxpr.jaxpr,
                                    ctx.name_stack, ctx.tokens_in, consts,
                                    *args, dim_var_values=ctx.dim_var_values)
    ctx.set_tokens_out(tokens)
    return out
mlir.register_lowering(custom_elemental_call_p, _custom_elemental_call_mlir_translation)


@custom_api_util.register_custom_decorator_type
class custom_elemental:
    fn: Callable
    nondiff_argnums: Sequence[int]
    elemental_partial: Callable
    
    def __init__(self, fn):
        self.fn = fn
        self.nondiff_argnums = None # nondiff_argnums
        self.elemental_partial = lambda x: x
        
    def defelemental(self, elemental_partial):
        self.elemental_partial = elemental_partial
        return elemental_partial

    def __call__(self, *args, **kwargs):
        f_, dyn_args = lu.wrap_init(self.fn), args
        elemental_ = lu.wrap_init(self.elemental_partial)
        out_flat = custom_elemental_call_p.bind(f_, elemental_, *dyn_args, symbolic_zeros=False)
        return out_flat

