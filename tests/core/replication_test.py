import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import SparseTensor, SparseDimension, DenseDimension

class TestReplicationMul(unittest.TestCase): 
    ### Replication tests
    def test_simple_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        _x = jnp.expand_dims(x, 1)
        _x = jnp.tile(_x, (1, 3))
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
        res = _x @ _y
        
        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, None)], x)
        sty = SparseTensor([DenseDimension(0, 3, None)], [DenseDimension(1, 2, 0)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))
        
    def test_replication_2d(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4,))
        _x = jnp.eye(4) * x
        _x = jnp.expand_dims(_x, 2)
        _x = jnp.tile(_x, (1, 1, 5))
        
        y = jrand.normal(ykey, (4, 5))
        _y = jnp.einsum("ij,jk->ijk", y, jnp.eye(5))


        res = jnp.einsum("ijk,jkl->il", _x, _y)
        
        stx = SparseTensor([SparseDimension(0, 4, 0, 1)], 
                        [SparseDimension(1, 4, 0, 0), DenseDimension(2, 5, None)], x)
        sty = SparseTensor([DenseDimension(0, 4, 0), SparseDimension(1, 5, 1, 2)], 
                        [SparseDimension(2, 5, 1, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_replication_2d_2nd(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4, 5))
        _x = jnp.einsum("ij,ik->ikj", x, jnp.eye(4))
        
        y = jrand.normal(ykey, (5,))
        _y = jnp.eye(5) * y
        _y = jnp.expand_dims(_y, 0)
        _y = jnp.tile(_y, (4, 1, 1))

        res = jnp.einsum("ijk,jkl->il", _x, _y)
        
        stx = SparseTensor([SparseDimension(0, 4, 0, 1)], 
                        [SparseDimension(1, 4, 0, 0), DenseDimension(2, 5, 1)], x)
        sty = SparseTensor([DenseDimension(0, 4, None), SparseDimension(1, 5, 0, 2)], 
                        [SparseDimension(2, 5, 0, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_4d_replication(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,jl->ijlk", x, d)
        
        y = jrand.normal(ykey, (5, 2))
        _y = jnp.expand_dims(y, 0)
        _y = jnp.tile(_y, (4, 1, 1))
        res = jnp.einsum("ijkl,klm->ijm", _x, _y)
                
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 4, 1, 2)], 
                        [SparseDimension(2, 4, 1, 1), DenseDimension(3, 5, 2)], x)
        sty = SparseTensor([DenseDimension(0, 4, None), DenseDimension(1, 5, 0)], 
                        [DenseDimension(2, 2, 1)], y)
        stres = stx * sty 
        
        iota = jnp.eye(5)
        
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
if __name__ == "__main__":
    unittest.main()