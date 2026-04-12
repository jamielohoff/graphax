# Graphax Performance Optimization Plan

**Goal**: Close the performance gap between `graphax.jacve` and `jax.jacrev` by eliminating unnecessary dense tensor materializations, exploiting algebraic structure (Kronecker deltas, identity matrices, scalar sparsity), and reducing Python-level overhead during vertex elimination.

JAX's `jacrev` works via symbolic VJP rules compiled to a single XLA kernel. Graphax builds a Python-level computational graph and interprets it with Python control flow — every structural inefficiency manifests as extra JAX ops or Python overhead.

---

## Issue 1 — `add`/`sub`/`neg` elemental rules materialize full arrays of ones

**Files**: `src/graphax/primitives/math.py` lines 12, 53–55, 62–64

**Root Cause**

```python
defelemental(lax.neg_p, lambda x: -jnp.ones_like(x))

def add_elemental_rule(x, y):
    return (jnp.ones_like(y), jnp.ones_like(x))  # 2 traced JAX ops

def sub_elemental_rule(x, y):
    return (jnp.ones_like(y), -jnp.ones_like(x))  # 2 traced JAX ops
```

The partial Jacobian of `add`/`sub` w.r.t. each input is semantically the identity: `∂(x+y)/∂x = 1`. For a vector of size `n`, this is a Kronecker delta — zero information, zero FLOPs to represent. Instead, `jnp.ones_like(x)` materializes a length-`n` array. This array flows into `make_parallel_jacobian` (base.py:81), which checks `elemental.size == 1` to decide between the pure-Kronecker path (no array) and the array-backed path. Because the array has size `n > 1`, it takes the array path, creating a `SparseTensor` with a full ones-array — blocking all `None`-val fast paths during vertex elimination.

**Fix**

```python
defelemental(lax.neg_p, lambda x: -1.0)

def add_elemental_rule(x, y):
    return (1.0, 1.0)
defelemental(lax.add_p, with_type_promotion(add_elemental_rule))

def sub_elemental_rule(x, y):
    return (1.0, -1.0)
defelemental(lax.sub_p, with_type_promotion(sub_elemental_rule))
```

`make_parallel_jacobian` at base.py:81–84 already handles the scalar case — when `type(elemental) is float or elemental.size == 1`, it sets `val_dim_fn = lambda _: None`, producing a `SparseTensor` with `val = 1.0` and all `SparseDimension`s having `val_dim=None`. This is the pure Kronecker-delta representation — no array at all.

**Expected Impact**: **High**. Every `add`/`sub`/`neg` node saves 1–2 full-array materializations during graph build, and enables identity fast-paths during vertex elimination. A transformer MLP block has O(L) such nodes.

---

## Issue 2 — `_mul` has no identity-tensor fast path

**File**: `src/graphax/sparse/tensor.py` lines 354–370

**Root Cause**

After an `add`/`sub`/`neg` vertex is eliminated, the result propagates a `SparseTensor` that is a pure Kronecker-delta. However `_mul` always routes through one of three full multiplication functions — `_pure_dot_product_mul`, `_pure_broadcast_mul`, or `_mixed_mul`. `_pure_broadcast_mul` handles `val=None` only when *both* operands have `val=None`. When one operand is the concrete edge (e.g., from a `dot_general`) and the other is the identity from `add`, the code falls into the `else` branch and runs `_swap_axes` + `_pad_tensors` + element-wise multiply. For an identity tensor, this is `x * ones = x`, computed via several reshapes and a broadcast multiply.

**Fix**

Add identity-detection fast paths at the top of `_mul` (before the `copy()` calls):

```python
def _is_identity_tensor(t: SparseTensor) -> bool:
    """True when t is a scaled Kronecker delta — no materialized data array."""
    return (
        all(isinstance(d, SparseDimension) and d.val_dim is None for d in t.dims)
        and (t.val is None or isinstance(t.val, (int, float))
             or (hasattr(t.val, 'size') and t.val.size == 1))
        and not t.pre_transforms
        and not t.post_transforms
    )

def _get_identity_scale(t: SparseTensor):
    return 1.0 if t.val is None else float(t.val)
```

In `_mul`, after the scalar-scalar shortcut:

```python
if _is_identity_tensor(lhs):
    scale = _get_identity_scale(lhs)
    res = rhs.copy(rhs.val if scale == 1.0 else rhs.val * scale)
    return res
if _is_identity_tensor(rhs):
    scale = _get_identity_scale(rhs)
    res = lhs.copy(lhs.val if scale == 1.0 else lhs.val * scale)
    return res
```

The `-1.0` case from `sub_elemental_rule` just needs a scalar scale — no JAX ops emitted.

**Expected Impact**: **Very High** (depends on Issue 1). Every `add`/`sub`/`neg` vertex elimination collapses to a scalar scale or no-op, instead of going through `_swap_axes` → `_pad_tensors` → element-wise multiply. In `y = tanh(W @ x + b)`, the `add` elimination touching the `tanh`-edge becomes free.

---

## Issue 3 — `reshape` and `slice` transforms unconditionally call `.dense(iota)`

**File**: `src/graphax/primitives/transforms.py` lines 123–142, 144–159, 186–205, 207–235

**Root Cause**

```python
def reshape_transform(pre, iota):
    # NOTE: Implement sparsity-aware version for significant speedup!
    full_val = pre.dense(iota)  # O(n²) materialization of full Jacobian matrix
```

`pre.dense(iota)` (tensor.py:122) builds a Kronecker product identity then does a reshape + element-wise multiply. For a vector of size `n`, this materializes an `(n, n)` dense matrix. In a transformer with embedding dimension 512, this is 262,144 float32 elements (1 MB) per reshape/slice operation — and attention blocks use many reshapes for head splitting/merging.

The TODO comment at line 121 explicitly acknowledges this: `"Implement sparsity-aware version for significant speedup!"`. The same problem exists in `inverse_reshape_transform` (line 145) and all slice transforms.

When the incoming `SparseTensor` is a pure diagonal (all `SparseDimension`s with `val_dim=None`), the Jacobian of `reshape` is still diagonal — just with relabeled output shape. There is no need to materialize a full dense matrix.

**Fix**

Add a sparsity fast path in `reshape_transform`:

```python
def reshape_transform(pre, iota):
    # Sparsity fast path: if pre is a diagonal Kronecker tensor, just relabel shapes
    if (pre.val is None or isinstance(pre.val, (int, float)) or
            (hasattr(pre.val, 'size') and pre.val.size == 1)):
        if all(isinstance(d, SparseDimension) and d.val_dim is None
               for d in pre.out_dims):
            l = len(val_out.shape)
            new_out_dims = [SparseDimension(j, s, None, l + j)
                            for j, s in enumerate(val_out.shape)]
            new_primal_dims = [SparseDimension(l + j, d.size, None, j)
                               for j, d in enumerate(pre.primal_dims)]
            return SparseTensor(new_out_dims, new_primal_dims, pre.val)
    # General fallback
    full_val = pre.dense(iota)
    ...
```

Apply the analogous pattern to `inverse_reshape_transform`, `slice_transform`, and `inverse_slice_transform`.

**Expected Impact**: **High** for any model with `jnp.reshape`, `jnp.ravel`, or `.flatten()`. The multi-head attention block in the test suite (`efficient_multihead_softmax_attention`) has explicit reshape operations for head splitting. This removes O(n²) dense materialization per reshape node.

---

## Issue 4 — `_sparse_add` creates fresh `jnp.eye` per call without using the global `iota`

**Files**: `src/graphax/sparse/tensor.py` lines 1387–1393; `src/graphax/sparse/utils.py` lines 33–59

**Root Cause**

```python
# tensor.py:1387–1393
if sum(_lshape) > len(_lshape):
    iota = eye_like(_lshape, len(lhs.out_dims))  # allocates new jnp.eye
    lhs_val = iota * lhs_val
```

`eye_like` at utils.py:33 creates fresh `jnp.eye` arrays every call — it does **not** use the global `iota` passed through the elimination process. This is called inside `_sparse_add` → `_add` → `__add__` → `_eliminate_vertex` line 290 every time two paths converge. Every graph node with fan-in > 1 triggers this, and for models with residual connections (transformers use `x + attn(...)`) this is hot.

Additionally, `SparseTensor.dense()` calls `eye_like_copy` twice (lines 153 and 171) with the same `eye_shape` argument — once for the identity test and once for tiling. These could be unified.

**Fix**

1. Pass the global `iota` through `_sparse_add` (add an `iota` parameter) so `eye_like` reuses the pre-allocated matrix.
2. Add a `functools.lru_cache` to `eye_like_copy` keyed on `(eye_shape_tuple, n_out_dims)` — within a single `jacve` call, `iota` is the same Python object, so `id(iota)` is stable.
3. In `dense()`, compute `eye_like_copy` once and reuse the result for both line 153 and line 171.

**Expected Impact**: **Medium-High**. Eliminates repeated `jnp.eye` allocations for every fan-in vertex and for every `dense()` call.

---

## Issue 5 — 3–4 redundant `SparseTensor.copy()` calls per edge pair in `_eliminate_vertex`

**File**: `src/graphax/core.py` lines 222–234

**Root Cause**

```python
post_val = _post_raw.copy()         # copy 1: outer loop, even if no transforms
for in_edge in ...:
    pre_val = _pre_raw.copy()       # copy 2: per inner iteration
    _pre_val = pre_val.copy()       # copy 3: defensive copy before transforms
    _post_val = post_val.copy()     # copy 4: defensive copy before transforms
```

`SparseTensor.copy()` does `copy.deepcopy(self.out_dims)` and `copy.deepcopy(self.primal_dims)` — deepcopying dataclass instances per copy call. With `n_in_edges × n_out_edges` iterations per vertex, this is O(degree²) deepcopy calls. Copies 3 and 4 are made only to be passed to `unload_post_transforms`/`unload_pre_transforms`, which call `.copy()` again internally (core.py:155, 163).

**Fix**

```python
# Only copy when transforms actually need to mutate the tensor
post_val = _post_raw  # no copy at outer level
for in_edge in ...:
    pre_val = _pre_raw  # no copy here
    if pre_val.post_transforms or post_val.pre_transforms:
        _post_val = post_val.copy()
        _pre_val = pre_val.copy()
        # apply transforms
    else:
        _post_val = post_val
        _pre_val = pre_val
```

Have `unload_post_transforms` and `unload_pre_transforms` own their own copy when needed.

**Expected Impact**: **Medium**. Reduces Python GC pressure from O(4·degree²·ndim) deepcopy operations per vertex to near-zero in the common case where transforms are absent.

---

## Issue 6 — `make_jaxpr` re-invoked on every call to the differentiated function

**File**: `src/graphax/core.py` line 91

**Root Cause**

```python
closed_jaxpr = jax.make_jaxpr(fun)(*flattened_args, **kwargs)
```

Called every time `jacfun(*args)` is invoked. `make_jaxpr` traces the Python function to symbolic form, running all Python code with abstract tracers. In eager mode (all unit tests run without JIT), this is paid directly as wall time. The recursive `jit_p` path (core.py:688–713) calls `vertex_elimination_jaxpr` for every `jit_p` node per `jacve` call — re-tracing sub-computations each time.

**Fix**

Add a shape-keyed cache inside `jacve`:

```python
def jacve(fun, order, argnums=(0,), ...):
    _jaxpr_cache: dict = {}

    @wraps(fun)
    def jacfun(*args, **kwargs):
        flattened_args, in_tree = jtu.tree_flatten(args)
        cache_key = (
            in_tree,
            tuple(
                (a.shape, a.dtype) if hasattr(a, 'shape') else type(a)
                for a in flattened_args
            ),
        )
        if cache_key not in _jaxpr_cache:
            _jaxpr_cache[cache_key] = jax.make_jaxpr(fun)(*flattened_args, **kwargs)
        closed_jaxpr = _jaxpr_cache[cache_key]
        ...
```

Within `jax.jit`, the outer JIT handles this already, so the cache mainly benefits eager execution and the recursive `jit_p` sub-computations.

**Expected Impact**: **Medium**. Eliminates O(#jit_subgraphs) re-trace calls for functions using `jax.jit` internally; significant in eager testing.

---

## Issue 7 — `eye_like_copy` recomputes Kronecker products for identical shapes

**File**: `src/graphax/sparse/utils.py` lines 62–122

**Root Cause**

The developer's own comment at line 121: `"# NOTE: This thing is crazy expensive to compute and not always necessary?"`.

`eye_like_copy` builds a Kronecker-product identity matrix by constructing `len(out_shape)` separate 2D `jnp.eye` arrays and multiplying in a loop. Called from:
- `SparseTensor.dense()` at tensor.py:153 and **again** at line 171 — called **twice** per materialization with the same args
- `unload_post_transforms`/`unload_pre_transforms` in core.py during elimination

Within a single `jacve` call, the same eye shape is constructed dozens of times. Since `iota` is computed once (core.py:609) and constant for the entire call, the result can be cached.

**Fix**

```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _eye_like_copy_cached(eye_shape_tuple: tuple, n_out_dims: int, iota_id: int, iota):
    return _eye_like_copy_impl(list(eye_shape_tuple), n_out_dims, iota)

def eye_like_copy(eye_shape, n_out_dims, iota):
    return _eye_like_copy_cached(tuple(eye_shape), n_out_dims, id(iota), iota)
```

Additionally, in `dense()`, compute `eye_like_copy` once and reuse the result for both occurrences.

**Expected Impact**: **Medium**. Repeated materializations of tensors with identical sparsity structure go from O(ndim × eye_alloc) to O(1) cache lookup.

---

## Issue 8 — `_replicate_along_axis` and `_sparse_add` use `jnp.tile` instead of broadcast

**File**: `src/graphax/sparse/tensor.py` lines 932, 950, 1382–1385

**Root Cause**

```python
# tensor.py:932, 950 (_replicate_along_axis)
lhs = jnp.expand_dims(lhs, axis=insert_axis)
lhs = jnp.tile(lhs, tiling)  # physical copy

# tensor.py:1382–1385 (_sparse_add)
if sum(ltiling) > len(ltiling):
    lhs_val = jnp.tile(lhs_val, ltiling)  # physical copy
```

`jnp.tile` is a full array replication — it creates an array `size × n` times larger. JAX's own `jacrev` VJP rules handle broadcasted additions via `lax.broadcast_in_dim` (semantically a no-copy view), not tile. `lax.dot_general` natively handles batched dimensions without physical replication.

**Fix**

For `_replicate_along_axis`, replace `expand_dims + tile` with `lax.broadcast_in_dim`, then pass the expanded dimension as a batch dimension in `lax.dot_general`. For `_sparse_add`, replace `jnp.tile` with `jnp.broadcast_to` for 1D broadcast cases, or `lax.broadcast_in_dim` with appropriate shape.

**Expected Impact**: **Medium-High** for models with batch dimensions or broadcasted additions (common in multi-head attention). Removes O(batch_size × head_dim) element copies per operation.

---

## Issue 9 — `_assert_sparse_tensor_consistency` runs unconditionally in the hot path

**Files**: `src/graphax/sparse/tensor.py` lines 255–259; `src/graphax/core.py` ~7 call sites

**Root Cause**

`_assert_sparse_tensor_consistency` runs Python loops over `st.dims` on every write to the graph — 7 explicit call sites in `_eliminate_vertex`. For a large graph (100-node function) with high fan-in/fan-out, this executes thousands of times per `jacve` call.

**Fix**

Gate all calls behind a debug flag:

```python
_GRAPHAX_DEBUG = os.environ.get('GRAPHAX_DEBUG', '0') == '1'

def _assert_sparse_tensor_consistency(st):
    if not _GRAPHAX_DEBUG:
        return
    ...
```

**Expected Impact**: **Medium**. For a transformer with ~100 equations and average tensor ndim ~4, this removes ~800 validation loop iterations per `jacve` invocation.

---

## Issue 10 — `iota` may be too small for intermediate tensors, causing fallback allocations

**Files**: `src/graphax/core.py` lines 123–151; `src/graphax/sparse/utils.py` lines 89–90

**Root Cause**

```python
# core.py:150–151
return jnp.eye(max(largest_output, largest_input), largest_input)

# utils.py:89–90 — fallback when iota is undersized
if iota.shape[0] < out_size or iota.shape[1] < primal_size:
    iota = jnp.eye(max(out_size, primal_size))  # fresh allocation
```

`_iota_shape` computes a global identity matrix sized to the largest *input/output* tensor. For encoder-decoder models, intermediate attention matrices `(seq_len × seq_len)` may be larger than parameter tensors. When the global `iota` is undersized for these intermediates, every `dense()` call for those edges allocates a fresh matrix.

**Fix**

In `_iota_shape`, scan all intermediate equation output variables — not just `invars` and `outvars`:

```python
all_vars = list(jaxpr.invars) + [
    ov for eqn in jaxpr.eqns for ov in eqn.outvars
    if isinstance(ov, core.Var)
]
largest = get_largest_tensor(all_vars)
```

**Expected Impact**: **Low-Medium** for simple functions; **Medium** for attention mechanisms.

---

## Issue 11 — `select_n` elemental rule allocates a full dense zero matrix

**File**: `src/graphax/primitives/reductions.py` lines 16–23

**Root Cause**

```python
jacsize = (size, size)
jacval = jnp.zeros(jacsize)  # allocates full size×size zero matrix
return [SparseTensor(new_out_dims, new_primal_dims, jacval) for _ in range(num_cases)]
```

`lax.select_n` is the conditional primitive underlying `jnp.where`. Its Jacobian for each non-selected case is zero. The correct representation is `val=0.0` (scalar) — the same scalar-val path already used by other elemental rules.

**Fix**

```python
jacval = 0.0  # scalar zero — no array allocation
return [SparseTensor(new_out_dims, new_primal_dims, jacval) for _ in range(num_cases)]
```

**Expected Impact**: **Low-Medium**. Every `jnp.where`, `jnp.clip`, ReLU (gelu via `jnp.where`) saves one `n×n` zero allocation per activation node.

---

## Summary Table

| # | Issue | File | Lines | Category | Impact |
|---|-------|------|-------|----------|--------|
| 1 | `add`/`sub`/`neg` materialize ones arrays | `primitives/math.py` | 12, 53–55, 62–64 | Dense tensor vs. identity | **High** |
| 2 | No identity fast-path in `_mul` | `sparse/tensor.py` | 354–370 | Ignores Kronecker delta structure | **High** (depends on #1) |
| 3 | `reshape`/`slice` transforms call `.dense(iota)` | `primitives/transforms.py` | 123–235 | O(n²) dense materialization | **High** |
| 4 | `_sparse_add` creates fresh `jnp.eye` per call | `sparse/tensor.py` | 1387–1393 | Redundant identity allocation | **Medium-High** |
| 5 | 3–4 redundant `SparseTensor.copy()` per edge pair | `core.py` | 222–234 | Unnecessary deepcopy overhead | **Medium** |
| 6 | `make_jaxpr` re-invoked every call | `core.py` | 91 | No structural reuse | **Medium** |
| 7 | `eye_like_copy` not cached | `sparse/utils.py` | 62–122 | Kronecker products recomputed | **Medium** |
| 8 | `_replicate_along_axis` uses `jnp.tile` instead of broadcast | `sparse/tensor.py` | 932, 950, 1382–1385 | Physical copy vs. view | **Medium-High** |
| 9 | `_assert_sparse_tensor_consistency` in hot path | `sparse/tensor.py` | 205–259 | Validation overhead | **Medium** |
| 10 | `iota` undersized for intermediate tensors | `core.py` | 123–151 | Fallback `jnp.eye` allocations | **Low-Medium** |
| 11 | `select_n` allocates full zero matrix | `primitives/reductions.py` | 16–23 | Dense tensor vs. zero scalar | **Low-Medium** |

## Recommended Implementation Order

1. **Issues 1 + 2 together** — `add`/`sub`/`neg` return scalar elementals, then `_mul` gets the identity fast-path. These two are tightly coupled and together eliminate the most unnecessary computation.
2. **Issue 3** — Independent, high-impact for deep learning workloads with reshape operations.
3. **Issue 11** — Trivial one-line fix; unlocks scalar zero fast-paths throughout.
4. **Issues 4 + 7 together** — Both address identity-matrix reuse; combine into one refactor of `eye_like_copy` and its callers.
5. **Issue 8** — Replace `jnp.tile` with `broadcast_in_dim`; requires care to preserve correctness.
6. **Issue 5** — Reduce `copy()` calls; straightforward refactor of `_eliminate_vertex`.
7. **Issue 9** — Add debug flag; trivial and safe.
8. **Issue 6** — Add `make_jaxpr` cache; medium complexity, medium benefit.
9. **Issue 10** — Extend `_iota_shape` to scan intermediates; low risk, low-medium benefit.
