import jax.lax as lax

from .base import elemental_rules, elemental_only_rules


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
