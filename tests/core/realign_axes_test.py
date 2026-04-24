"""
Tests for _realign_axes in sparse tensor algebra.

SparseIndex always occurs in pairs: one in out_dims, one in primal_dims.
`id` = position in st.dims (out_dims + primal_dims).
`other_id` = position of partner in st.dims.
Both members of a pair share the same `size` and `val_axis`.

_realign_axes is called after SparseIndex val_axis is set to None because
val.shape[val_axis] == 1 while d.size > 1. The squeeze removes the size-1
axis and all remaining val_axis values must shift down accordingly.
"""
import unittest

import jax.numpy as jnp

from graphax.sparse.tensor import (
    DenseIndex,
    SparseIndex,
    SparseTensor,
    _realign_axes,
)


class RealignAxesTests(unittest.TestCase):
    """Tests for _realign_axes."""

    def test_squeeze_single_sparse_pair_axis(self):
        """Squeeze SparseIndex pair's size-1 axis; remaining DenseIndices shift down."""
        # st.dims = [SparseIndex(0), DenseIndex(1), DenseIndex(2), SparseIndex(3)]
        # pair (0,3): id=0/3, other_id=3/0, val_axis=0, size=2, val.shape[0]=1 (squeeze target)
        # DenseIndex(1): val_axis=1, size=3, val.shape[1]=3
        # DenseIndex(2): val_axis=2, size=4, val.shape[2]=4
        out_dims = (SparseIndex(0, 2, 0, 3), DenseIndex(1, 3, 1))
        primal_dims = (DenseIndex(2, 4, 2), SparseIndex(3, 2, 0, 0))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((1, 3, 4)))

        # Simulate pre-conditions: SparseIndex pair val_axis set to None
        st.out_dims[0].val_axis = None
        st.primal_dims[1].val_axis = None

        _realign_axes(st, [0])

        self.assertEqual(st.val.shape, (3, 4))
        # DenseIndex(1): was 1, squeezed axes < 1: [0] => 1 decrement => 0
        self.assertEqual(st.out_dims[1].val_axis, 0)
        # DenseIndex(2): was 2, squeezed axes < 2: [0] => 1 decrement => 1
        self.assertEqual(st.primal_dims[0].val_axis, 1)

    def test_squeeze_among_two_sparse_pairs(self):
        """Two SparseIndex pairs; squeeze one pair's axis, the other pair shifts."""
        # st.dims = [SparseIndex(0), SparseIndex(1), DenseIndex(2),
        #            SparseIndex(3), DenseIndex(4), SparseIndex(5)]
        # pair (0,3): id=0/3, other_id=3/0, val_axis=0, size=2, val.shape[0]=1 (squeezed)
        # pair (1,5): id=1/5, other_id=5/1, val_axis=1, size=3, val.shape[1]=3
        # DenseIndex(2): val_axis=2, size=5
        # DenseIndex(4): val_axis=3, size=4
        out_dims = (SparseIndex(0, 2, 0, 3), SparseIndex(1, 3, 1, 5), DenseIndex(2, 5, 2))
        primal_dims = (SparseIndex(3, 2, 0, 0), DenseIndex(4, 4, 3), SparseIndex(5, 3, 1, 1))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((1, 3, 5, 4)))

        # Squeeze the first SparseIndex pair
        st.out_dims[0].val_axis = None
        st.primal_dims[0].val_axis = None

        _realign_axes(st, [0])

        self.assertEqual(st.val.shape, (3, 5, 4))
        # SparseIndex(1): was 1, squeezed axes < 1: [0] => 1 => 0
        self.assertEqual(st.out_dims[1].val_axis, 0)
        # DenseIndex(2): was 2, squeezed axes < 2: [0] => 1 => 1
        self.assertEqual(st.out_dims[2].val_axis, 1)
        # DenseIndex(4): was 3, squeezed axes < 3: [0] => 1 => 2
        self.assertEqual(st.primal_dims[1].val_axis, 2)
        # SparseIndex(5): was 1, squeezed axes < 1: [0] => 1 => 0
        self.assertEqual(st.primal_dims[2].val_axis, 0)

    def test_squeeze_axis_above_all_val_axes(self):
        """Squeeze axis higher than all remaining val_axis; no decrements."""
        # st.dims = [DenseIndex(0), DenseIndex(1), SparseIndex(2),
        #            DenseIndex(3), SparseIndex(4)]
        # pair (2,4): id=2/4, other_id=4/2, val_axis=3, size=4, val.shape[3]=1 (squeezed)
        # DenseIndex(0): val_axis=0, size=2
        # DenseIndex(1): val_axis=1, size=3
        # DenseIndex(3): val_axis=2, size=5
        out_dims = (DenseIndex(0, 2, 0), DenseIndex(1, 3, 1), SparseIndex(2, 4, 3, 4))
        primal_dims = (DenseIndex(3, 5, 2), SparseIndex(4, 4, 3, 2))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((2, 3, 5, 1)))

        st.out_dims[2].val_axis = None
        st.primal_dims[1].val_axis = None

        _realign_axes(st, [3])

        self.assertEqual(st.val.shape, (2, 3, 5))
        # DenseIndex(0): was 0, squeezed axes < 0: [] => 0 => 0
        self.assertEqual(st.out_dims[0].val_axis, 0)
        # DenseIndex(1): was 1, squeezed axes < 1: [] => 0 => 1
        self.assertEqual(st.out_dims[1].val_axis, 1)
        # DenseIndex(3): was 2, squeezed axes < 2: [] => 0 => 2
        self.assertEqual(st.primal_dims[0].val_axis, 2)

    def test_squeeze_axis_between_val_axes(self):
        """Squeeze axis between two val_axis values; only higher ones shift."""
        # st.dims = [DenseIndex(0), SparseIndex(1), SparseIndex(2),
        #            DenseIndex(3), DenseIndex(4)]
        # pair (1,2): id=1/2, other_id=2/1, val_axis=2, size=3, val.shape[2]=1 (squeezed)
        # DenseIndex(0): val_axis=0, size=2
        # DenseIndex(3): val_axis=1, size=5
        # DenseIndex(4): val_axis=3, size=6
        out_dims = (DenseIndex(0, 2, 0), SparseIndex(1, 3, 2, 2))
        primal_dims = (SparseIndex(2, 2, 2, 1), DenseIndex(3, 5, 1), DenseIndex(4, 6, 3))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((2, 5, 1, 6)))

        st.out_dims[1].val_axis = None
        st.primal_dims[0].val_axis = None

        _realign_axes(st, [2])

        self.assertEqual(st.val.shape, (2, 5, 6))
        # DenseIndex(0): was 0, squeezed axes < 0: [] => 0 => 0
        self.assertEqual(st.out_dims[0].val_axis, 0)
        # DenseIndex(3): was 1, squeezed axes < 1: [] => 0 => 1
        self.assertEqual(st.primal_dims[1].val_axis, 1)
        # DenseIndex(4): was 3, squeezed axes < 3: [2] => 1 => 2
        self.assertEqual(st.primal_dims[2].val_axis, 2)

    def test_squeeze_multiple_axes_mixed_positions(self):
        """Squeeze axes at positions 0 and 2; correct decrement counts."""
        # st.dims = [SparseIndex(0), DenseIndex(1), SparseIndex(2),
        #            SparseIndex(3), SparseIndex(4), DenseIndex(5)]
        # pair (0,3): val_axis=0, size=2, val.shape[0]=1 (squeezed)
        # pair (2,4): val_axis=2, size=4, val.shape[2]=1 (squeezed)
        # DenseIndex(1): val_axis=1, size=3
        # DenseIndex(5): val_axis=3, size=5
        out_dims = (SparseIndex(0, 2, 0, 3), DenseIndex(1, 3, 1), SparseIndex(2, 4, 2, 4))
        primal_dims = (SparseIndex(3, 2, 0, 0), SparseIndex(4, 4, 2, 2), DenseIndex(5, 5, 3))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((1, 3, 1, 5)))

        st.out_dims[0].val_axis = None
        st.primal_dims[0].val_axis = None
        st.out_dims[2].val_axis = None
        st.primal_dims[1].val_axis = None

        _realign_axes(st, [0, 2])

        self.assertEqual(st.val.shape, (3, 5))
        # DenseIndex(1): was 1, squeezed axes < 1: [0] => 1 => 0
        self.assertEqual(st.out_dims[1].val_axis, 0)
        # DenseIndex(5): was 3, squeezed axes < 3: [0, 2] => 2 => 1
        self.assertEqual(st.primal_dims[2].val_axis, 1)

    def test_empty_squeeze_list(self):
        """Empty squeeze list leaves everything unchanged."""
        # pair (1,2) shares val_axis=1 with val.shape[1]=1 (broadcast case)
        out_dims = (DenseIndex(0, 2, 0), SparseIndex(1, 3, 1, 2))
        primal_dims = (SparseIndex(2, 2, 1, 1), DenseIndex(3, 4, 2))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((2, 1, 4)))

        _realign_axes(st, [])

        self.assertEqual(st.val.shape, (2, 1, 4))
        self.assertEqual(st.out_dims[0].val_axis, 0)
        self.assertEqual(st.out_dims[1].val_axis, 1)
        self.assertEqual(st.primal_dims[0].val_axis, 1)
        self.assertEqual(st.primal_dims[1].val_axis, 2)

    def test_none_val_axis_unchanged(self):
        """Indices with val_axis=None are never modified."""
        # pair (0,3): val_axis=None, size=2
        # DenseIndex(1): val_axis=0, size=3
        # DenseIndex(2): val_axis=1, size=4
        out_dims = (SparseIndex(0, 2, None, 3), DenseIndex(1, 3, 0))
        primal_dims = (DenseIndex(2, 4, 1), SparseIndex(3, 2, None, 0))
        st = SparseTensor(out_dims, primal_dims, jnp.ones((1, 4)))

        _realign_axes(st, [0])

        self.assertEqual(st.val.shape, (4,))
        # None stays None
        self.assertIsNone(st.out_dims[0].val_axis)
        self.assertIsNone(st.primal_dims[1].val_axis)
        # DenseIndex(1): was 0, squeezed axes < 0: [] => 0 => 0
        self.assertEqual(st.out_dims[1].val_axis, 0)
        # DenseIndex(2): was 1, squeezed axes < 1: [0] => 1 => 0
        self.assertEqual(st.primal_dims[0].val_axis, 0)


if __name__ == "__main__":
    unittest.main()
