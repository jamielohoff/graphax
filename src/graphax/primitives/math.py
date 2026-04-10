from typing import Callable

import jax
import jax.lax as lax
import jax.numpy as jnp

from .base import defelemental, defelemental2

Array = jax.Array


defelemental(lax.neg_p, lambda x: -jnp.ones_like(x))
defelemental2(
    lax.abs_p, lambda out, primal: primal / out
)  # NOTE: not differentiable here!
defelemental(lax.integer_pow_p, lambda x, y: y * lax.integer_pow(x, y - 1))

defelemental2(lax.exp_p, lambda out, primal: out)
defelemental(lax.log_p, lambda x, accuracy: 1.0 / x)
defelemental2(lax.sqrt_p, lambda out, primal, accuracy: 0.5 / out)
defelemental(lax.square_p, lambda x: 2.0 * x)
defelemental2(lax.logistic_p, lambda out, primal, accuracy: out * (1.0 - out))
defelemental(lax.log1p_p, lambda x, accuracy: 1.0 / (1.0 + x))

defelemental(lax.sin_p, lambda x, accuracy: lax.cos(x, accuracy=accuracy))
defelemental(lax.asin_p, lambda x: 1.0 / lax.sqrt(1.0 - x**2)) 
defelemental(lax.cos_p, lambda x, accuracy: -lax.sin(x, accuracy=accuracy))
defelemental(lax.acos_p, lambda x: -1.0 / lax.sqrt(1.0 - x**2))
defelemental2(lax.tan_p, lambda out, primal, accuracy: 1.0 + out**2)
defelemental(lax.atan_p, lambda x: 1.0 / (1.0 + x**2))

defelemental(lax.sinh_p, lax.cosh)
defelemental(lax.asinh_p, lambda x: lax.sqrt(1.0 + x**2))
defelemental(lax.cosh_p, lax.sinh)
defelemental(lax.acosh_p, lambda x: 1.0 / lax.sqrt(x**2 - 1.0))
defelemental2(lax.tanh_p, lambda out, primal, accuracy: 1.0 - out**2)
defelemental(lax.atanh_p, lambda x: 1.0 / (1.0 - x**2))

defelemental(lax.erf_p, lambda x: 2.0 * lax.exp(-(x**2)) / lax.sqrt(jnp.pi))


def with_type_promotion(fn: Callable) -> Callable:
    def promoted_fn(*operands, **params) -> tuple[Array, ...]:
        res = fn(*operands, **params)
        type = jnp.result_type(*(op.dtype for op in operands))
        return tuple(lax.convert_element_type(el, type) for el in res)

    return promoted_fn


# TODO this can be significantly optimized
# Currently we are creating a new array of ones everytime. Not smart!
@with_type_promotion
def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))


defelemental(lax.add_p, add_elemental_rule)


# TODO this can also be optimized significantly
@with_type_promotion
def sub_elemental_rule(x, y):
    return (jnp.ones_like(y), -jnp.ones_like(x))


defelemental(lax.sub_p, sub_elemental_rule)


@with_type_promotion
def mul_elemental_rule(x, y):
    return (y, x)


defelemental(lax.mul_p, mul_elemental_rule)


@with_type_promotion
def div_elemental_rule(x, y):
    return (1.0 / y, -x / y**2)


defelemental(lax.div_p, div_elemental_rule)


@with_type_promotion
def atan2_elemental_rule(x, y):
    abs2 = x**2 + y**2
    return (y / abs2, -x / abs2)


defelemental(lax.atan2_p, atan2_elemental_rule)


@with_type_promotion
def max_elemental_rule(x, y):
    return (x < y, x >= y)


defelemental(lax.max_p, max_elemental_rule)


@with_type_promotion
def min_elemental_rule(x, y):
    return (jnp.where(x < y, 1, 0), jnp.where(x < y, 0, 1))


defelemental(lax.min_p, min_elemental_rule)


@with_type_promotion
def eq_elemental_rule(x, y):
    return (jnp.zeros_like(y), jnp.zeros_like(x))


defelemental(lax.eq_p, eq_elemental_rule)
defelemental(lax.gt_p, eq_elemental_rule)
defelemental(lax.lt_p, eq_elemental_rule)


@with_type_promotion
def pow_elemental_rule(out, x, y):
    return (y * x ** (y - 1), jnp.log(x) * out)


defelemental2(lax.pow_p, pow_elemental_rule)
