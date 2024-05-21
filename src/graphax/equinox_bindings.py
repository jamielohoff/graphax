import functools as ft
from functools import wraps
from typing import Any, Callable, Union, Sequence

import jax
import jax.tree_util as jtu

from equinox import is_array
from equinox._filters import combine, partition, is_inexact_array
from equinox._module import Module, Partial, module_update_wrapper
from equinox._custom_types import sentinel
from equinox._deprecate import deprecated_0_10
from equinox._doc_utils import doc_remove_args
from equinox import filter_make_jaxpr

from .core import vertex_elimination_jaxpr


class _JacveWrapper(Module):
    _fun: Callable
    _gradkwargs: dict[str, Any]
    
    @property
    def __wrapped__(self):
        return self._fun
    
    def __call__(self, *args, **kwargs):
        def fun_jacve(_diff_x, _nondiff_x, *_args, **_kwargs):
            _x = combine(_diff_x, _nondiff_x)
            flat_x = jtu.tree_flatten(_x)
            _argnums = [i for i, xs in enumerate(flat_x[0]) if is_inexact_array(xs)]

            return eqx_jacve(self._fun, argnums=_argnums, **self._gradkwargs)(_x, *_args, **_kwargs)
        
        x, *args = args
        diff_x, nondiff_x = partition(x, is_inexact_array)
        
        return fun_jacve(diff_x, nondiff_x, *args, **kwargs)
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return Partial(self, instance)


@doc_remove_args("gradkwargs")
def filter_jacve(
    fun=sentinel, **gradkwargs
) -> Callable:
    """
    TODO docstring
    """

    if fun is sentinel:
        return ft.partial(filter_jacve, **gradkwargs)

    deprecated_0_10(gradkwargs, "arg")
    deprecated_0_10(gradkwargs, "filter_spec")
    argnums = gradkwargs.pop("argnums", None)
    if argnums is not None:
        raise ValueError(
            "`argnums` should not be passed. If you need to differentiate "
            "multiple objects then collect them into a tuple and pass that "
            "as the first argument."
        )

    return module_update_wrapper(_JacveWrapper(fun, gradkwargs), fun)


# TODO pytree crap needs overhauling
def eqx_jacve(fun: Callable, 
            order: Union[Sequence[int], str], 
            argnums: Sequence[int] = (0,),
            count_ops: bool = False,
            dense_representation: bool = True) -> Callable:
    @wraps(fun)
    def wrapped(*args, **kwargs):
        # TODO Make repackaging work properly with one input value only
        in_tree = jtu.tree_structure(args)
        closed_jaxpr, _, _ = filter_make_jaxpr(fun)(*args, **kwargs)
        print(closed_jaxpr.jaxpr)

        x, *args = args
        flattened_x, _ = jtu.tree_flatten(x)
        flattened_args, _ = jtu.tree_flatten(args)

        _x = [arg for arg in flattened_x if is_inexact_array(arg)]
        _args = [arg for arg in flattened_args if is_array(arg)]
        _args = _x + _args

        out = vertex_elimination_jaxpr(closed_jaxpr.jaxpr, 
                                        order, 
                                        closed_jaxpr.literals, 
                                        *_args, 
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
                _out = []
                i = 0
                for j, arg in enumerate(flattened_x + flattened_args):
                    if arg is not None and j in argnums:
                        _out.append(out[0][i])
                        i += 1
                    else:
                        _out.append(None)
                return jtu.tree_unflatten(in_tree, _out)[0]
            
            _out = jtu.tree_unflatten(in_tree, out)
            return _out
    return wrapped

