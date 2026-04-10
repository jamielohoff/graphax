import jax.lax as lax
import jax._src.lax.lax as lax_src

from .base import elemental_rules, elemental_only_rules, multi_output_elemental_only_rules
from .transforms import _slice_elementals


def iota_elemental_rule(primals, **params):
    val_out = lax.iota_p.bind(*primals, **params)
    return val_out, []


def iota_elemental_only(primal_out, primals, **params):
    return []


elemental_rules[lax.iota_p] = iota_elemental_rule
elemental_only_rules[lax.iota_p] = iota_elemental_only


def device_put_elemental_rule(primals, **params):
    val_out = lax.device_put_p.bind(*primals, **params)
    return val_out, []


def device_put_elemental_only(primal_out, primals, **params):
    return []


elemental_rules[lax.device_put_p] = device_put_elemental_rule
elemental_only_rules[lax.device_put_p] = device_put_elemental_only


def stop_gradient_elemental_rule(primals, **params):
    val_out = lax.stop_gradient_p.bind(*primals, **params)
    return val_out, []


def stop_gradient_elemental_only(primal_out, primals, **params):
    return []


elemental_rules[lax.stop_gradient_p] = stop_gradient_elemental_rule
elemental_only_rules[lax.stop_gradient_p] = stop_gradient_elemental_only


# ---------- split ----------

def _split_elemental_for_output(primal, primal_out_k, start_k, end_k, axis):
    """Elemental partial for the k-th output of split w.r.t. the input.

    This is identical to the lax.slice elemental with start/limit indices
    chosen to select the k-th chunk along the split axis.
    """
    ndim = primal.ndim
    start_indices = tuple(start_k if i == axis else 0 for i in range(ndim))
    limit_indices = tuple(end_k if i == axis else primal.shape[i] for i in range(ndim))
    slice_params = {
        'start_indices': start_indices,
        'limit_indices': limit_indices,
        'strides': None,
    }
    # _slice_elementals returns list[SparseTensor] with one entry per invar
    return _slice_elementals([primal], primal_out_k, **slice_params)


def split_elemental_only(primal_outs, primals, **params):
    """Multi-output elemental rule for lax.split_p.

    Returns elementals[outvar_idx][invar_idx].  split has one invar and N
    outvars, so the outer list has N entries, each a length-1 list.
    """
    primal = primals[0]
    sizes = params['sizes']
    axis = params['axis']

    result = []
    start = 0
    for size, primal_out_k in zip(sizes, primal_outs):
        end = start + int(size)
        result.append(_split_elemental_for_output(primal, primal_out_k, start, end, axis))
        start = end
    return result


multi_output_elemental_only_rules[lax_src.split_p] = split_elemental_only
