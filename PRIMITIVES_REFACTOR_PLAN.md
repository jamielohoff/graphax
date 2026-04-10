# Primitives Refactoring Plan

## Objectives

1. **Split** `src/graphax/primitives.py` (~1500 lines, one flat file) into a proper
   subpackage with focused modules.
2. **Deferred elemental dispatch**: for every primitive, `_build_graph` dispatches
   *only* the primal op (`primitive.bind`) eagerly.  All elemental partial
   computations are wrapped in `LazyEdge` thunks and only dispatched when
   `_eliminate_vertex` actually consumes the edge.  Pruned edges never dispatch
   their elemental ops.

---

## Step 0 — File Split (First Order of Business)

Convert `src/graphax/primitives.py` into a subpackage:

```
src/graphax/primitives/
    __init__.py       re-exports everything needed by the rest of graphax
    base.py           registration infrastructure
    math.py           scalar/unary/binary math (defelemental / defelemental2)
    linalg.py         dot_general
    reductions.py     reduce_sum, reduce_max, reduce_min, select_n
    structural.py     iota, device_put, stop_gradient
    transforms.py     JacobianTransform + all transform rules
    pjit.py           pjit_elemental_rule, _trace_subjaxpr
```

Move existing code verbatim — **no logic changes in this step**.  Run the full
test suite after the split to confirm nothing broke.

---

## Step 1 — Deferred Dispatch Infrastructure (`base.py`)

Add alongside `elemental_rules`:

```python
elemental_only_rules: dict[Primitive, Callable]
# signature: (primal_out, primals, **params) -> list[SparseTensor]
```

Add two new `standard_*_only` functions that mirror `standard_elemental` and
`standard_elemental2` but **take `primal_out` as the first argument instead of
calling `primitive.bind`**:

```python
def standard_elemental_only(elementalrule, primal_out, primals, **params):
    elementals = elementalrule(*primals, **_filter_params(elementalrule, params))
    ...  # same make_parallel_jacobian wrapping as standard_elemental

def standard_elemental2_only(elementalrule, primal_out, primals, **params):
    elementals = elementalrule(primal_out, *primals, **_filter_params(...))
    ...
```

Update `defelemental` and `defelemental2` to register in **both** dicts.

Update `_build_graph` in `core.py`:
- If `primitive in elemental_only_rules`: call `primitive.bind` for the primal,
  store `LazyEdge` thunks that call `elemental_only_rules[primitive](primal_out,
  snap, **params)`.  No elemental ops dispatched during `_build_graph`.
- Else (fallback for custom rules not yet split): call full `cce` once,
  pre-populate the elemental cache so thunks never re-call `cce`.

---

## Step 2 — `math.py`

**Zero additional work.**  All primitives here use `defelemental` or
`defelemental2`, which already register in `elemental_only_rules` after Step 1.

Primitives covered: `neg`, `abs`, `integer_pow`, `exp`, `log`, `sqrt`, `square`,
`logistic`, `log1p`, `sin`, `asin`, `cos`, `acos`, `tan`, `atan`, `atanh`,
`sinh`, `asinh`, `cosh`, `acosh`, `tanh`, `erf`, `add`, `sub`, `mul`, `div`,
`atan2`, `max`, `min`, `eq`, `gt`, `lt`, `pow`.

---

## Step 3 — `linalg.py` (`dot_general_p`)

**Benefit: HIGH** — elementals are `SparseTensor(..., val=rhs)` /
`SparseTensor(..., val=lhs)`.  They carry the *other* input as the value; the
primal output is **not used** in their construction.

Split:
```python
def dot_general_elemental_only(primal_out, primals, **params):
    lhs, rhs = primals
    # same dimension-number logic as dot_general_elemental_rule
    # build lhs_tensor, rhs_tensor — no primitive.bind call
    return [lhs_tensor, rhs_tensor]

elemental_only_rules[lax.dot_general_p] = dot_general_elemental_only
```

---

## Step 4 — `reductions.py` (`reduce_sum_p`, `reduce_max_p`, `reduce_min_p`, `select_n_p`)

**Benefit: HIGH** — elemental computations involve real JAX ops (`jnp.ones`,
`jnp.where`, `jnp.zeros`) that currently dispatch during `_build_graph`.

### `reduce_sum_p`
Elemental value is `jnp.ones(shape, dtype=float32)` — the primal output is only
used for `get_ndim(val_out)` (= `val_out.ndim`), which equals
`primal_out.ndim`.  Pass `primal_out` in and derive ndim from it.

### `reduce_max_p` / `reduce_min_p`
Elemental uses `jnp.where(primal == val_out, 1, 0)` — needs `val_out`.  This
follows the `defelemental2` pattern exactly: pass `primal_out` and use it
directly instead of re-calling `primitive.bind`.

### `select_n_p`
Elemental is `jnp.zeros(jacsize)` — independent of `val_out`.  Add trivial
elemental-only variant.

---

## Step 5 — `structural.py` (`iota_p`, `device_put_p`, `stop_gradient_p`)

**Benefit: NONE** — all three return `(val_out, [])`.  No elemental ops are
dispatched even today.  The elemental-only functions trivially return `[]`.
Register them in `elemental_only_rules` for completeness so `_build_graph` takes
the happy path (avoiding the fallback branch) for these primitives.

---

## Step 6 — `transforms.py` (`transpose_p`, `reshape_p`, `slice_p`, `broadcast_in_dim_p`, `squeeze_p`, `concatenate_p`, `convert_element_type_p`)

**Benefit: LOW for dispatch, HIGH for clarity.**

All transform rules return `SparseTensor(val=None, post_transforms=[...])`.
Because `val=None`, there are **no JAX array operations in the elemental
computation** — it is pure Python object construction (`JacobianTransform` +
`SparseTensor`).  The actual JAX ops (reshape, scatter, etc.) only run when the
transform is *applied* inside `_eliminate_vertex`.  Transforms are therefore
already deferred by design.

Split is still worth doing so `_build_graph` takes the clean happy path for all
transform primitives.  The elemental-only function simply constructs and returns
the transform-carrying `SparseTensor` without calling `primitive.bind`.

Note: `concatenate_p` returns one `SparseTensor` *per primal* (one per input
tensor).  The elemental-only function must return a list of length
`len(primals)`, mirroring the current rule.

---

## Step 7 — `pjit.py` (`jit_p`)

**Benefit: NONE** — `pjit_elemental_rule` returns `(outputs, [])`.  No elementals.
Elemental-only function returns `[]`.  Register for completeness.

`_trace_subjaxpr` is also moved here.  It is currently unused in the main
execution path (commented-out pjit approach) but kept for future work on proper
JIT-of-jacve composability (LAZY_EVAL_PLAN.md Milestone 7).

---

## Dispatch Benefit Summary

| Module | Elemental has JAX ops? | Dispatch benefit of split |
|---|---|---|
| `math.py` | Yes (cos, convert_type, etc.) | **High** |
| `linalg.py` | No (just input refs in SparseTensor) | Medium (avoids `bind` in thunk) |
| `reductions.py` | Yes (`ones`, `where`, `zeros`) | **High** |
| `structural.py` | No elementals | None (completeness only) |
| `transforms.py` | No (val=None, pure Python) | None (correctness/completeness) |
| `pjit.py` | No elementals | None (completeness only) |

---

## Execution Order

| Step | File(s) changed | Test gate |
|---|---|---|
| 0 | `primitives/` (split, no logic change) | All existing tests pass |
| 1 | `primitives/base.py`, `core.py` | All existing tests pass |
| 2 | `primitives/math.py` | No changes needed; covered by Step 1 |
| 3 | `primitives/linalg.py` | All existing tests pass |
| 4 | `primitives/reductions.py` | All existing tests pass |
| 5 | `primitives/structural.py` | All existing tests pass |
| 6 | `primitives/transforms.py` | All existing tests pass |
| 7 | `primitives/pjit.py` | All existing tests pass |

Each step is a self-contained commit.  After Step 1, the jaxpr of `Simple()`
should show **no `_` (DropVar) equations** for the math primitives.
