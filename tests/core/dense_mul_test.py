import unittest

import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import SparseTensor, DenseIndex

class TestDenseMul(unittest.TestCase):            
    def test_simple_matmul(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4, 3))
        y = jrand.normal(ykey, (3, 2))
        res = x @ y
        
        stx = SparseTensor([DenseIndex(0, 4, 0)], [DenseIndex(1, 3, 1)], x)
        sty = SparseTensor([DenseIndex(0, 3, 0)], [DenseIndex(1, 2, 1)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))
        
    def test_3d_dense_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        y = jrand.normal(ykey, (5, 2, 2))
        res = jnp.einsum("ijk,klm->ijlm", x, y)
        
        stx = SparseTensor([DenseIndex(0, 3, 0), DenseIndex(1, 4, 1), ], 
                        [DenseIndex(2, 5, 2)], x)
        sty = SparseTensor([DenseIndex(0, 5, 0)], 
                        [DenseIndex(1, 2, 1), DenseIndex(2, 2, 2)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))
        
    def test_3d_dense_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        y = jrand.normal(ykey, (4, 5, 2))
        res = jnp.einsum("ijk,jkl->il", x, y)
        
        stx = SparseTensor([DenseIndex(0, 3, 0)], 
                        [DenseIndex(1, 4, 1), DenseIndex(2, 5, 2)], x)
        sty = SparseTensor([DenseIndex(0, 4, 0), DenseIndex(1, 5, 1)], 
                        [DenseIndex(2, 2, 2)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.all(res == stres.val))
    
    def test_4d_dense_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4, 5, 6))
        y = jrand.normal(ykey, (5, 6, 2, 7))

        res = jnp.einsum("ijkl,klmn->ijmn", x, y)
            
        stx = SparseTensor([DenseIndex(0, 3, 0), DenseIndex(1, 4, 1)], 
                        [DenseIndex(2, 5, 2), DenseIndex(3, 6, 3)], x)
        sty = SparseTensor([DenseIndex(0, 5, 0), DenseIndex(1, 6, 1)], 
                        [DenseIndex(2, 2, 2), DenseIndex(3, 7, 3)], y)
        stres = stx * sty
                
        iota = jnp.eye(15)
                
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        

if __name__ == "__main__":
    unittest.main()
    
