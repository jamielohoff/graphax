import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import SparseTensor, SparseDimension, DenseDimension

class TestBroadcastMul(unittest.TestCase):            
    ### Tests for 2d tensors aka matrices
    def test_simple_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))
        print(_x.shape)
        y = jrand.normal(ykey, (3, 2))
        res = _x @ y
        
        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, None)], x)
        sty = SparseTensor([DenseDimension(0, 3, 0)], [DenseDimension(1, 2, 1)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))
        
    def test_double_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))

        y = jrand.normal(ykey, (2,))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (3, 1))
        print(_x.shape, _y.shape)
        res = _x @ _y
        
        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, None)], x)
        sty = SparseTensor([DenseDimension(0, 3, None)], [DenseDimension(1, 2, 0)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))
        

if __name__ == '__main__':
    unittest.main()