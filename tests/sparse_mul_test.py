import unittest

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand

from graphax.sparse.tensor import SparseTensor, SparseDimension, DenseDimension

class TestBroadcastMul(unittest.TestCase):            
    ### Tests for 2d tensors aka matrices
    def test_simple_matmul(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4, 3))
        y = jrand.normal(ykey, (3, 2))
        res = x @ y
        
        stx = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, 1)], x)
        sty = SparseTensor([DenseDimension(0, 3, 0)], [DenseDimension(1, 2, 1)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))

    def test_simple_broadcast(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        y = jrand.normal(ykey, (4,))
        res = jnp.diag(x) @ jnp.diag(y)
        
        stx = SparseTensor([SparseDimension(0, 4, 0, 1)], [SparseDimension(1, 4, 0, 0)], x)
        sty = SparseTensor([SparseDimension(0, 4, 0, 1)], [SparseDimension(1, 4, 0, 0)], y)
        stres = stx * sty
        
        iota = jnp.eye(4)
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_simple_sparse_dense(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (4,))
        y = jrand.normal(ykey, (4, 3))
        res = jnp.diag(x) @ y
        
        stx = SparseTensor([SparseDimension(0, 4, 0, 1)], [SparseDimension(1, 4, 0, 0)], x)
        sty = SparseTensor([DenseDimension(0, 4, 0)], [DenseDimension(1, 3, 1)], y)
        stres = stx * sty
        
        iota = jnp.eye(4)
        
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_simple_dense_sparse(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4))
        y = jrand.normal(ykey, (4,))
        res = x @ jnp.diag(y)
        
        stx = SparseTensor([DenseDimension(0, 3, 0)], [DenseDimension(1, 4, 1)], x)
        sty = SparseTensor([SparseDimension(0, 4, 0, 1)], [SparseDimension(1, 4, 0, 0)], y)
        stres = stx * sty
        
        iota = jnp.eye(4)

        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
    
    def test_simple_Nones(self):        
        _x = jnp.eye(3)
        _y = jnp.eye(3)

        res = _x @ _y
            
        stx = SparseTensor([SparseDimension(0, 3, None, 1)], 
                        [SparseDimension(1, 3, None, 0)], None)
        sty = SparseTensor([SparseDimension(0, 3, None, 1)], 
                        [SparseDimension(1, 3, None, 0)], None)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_simple_dense_None(self): 
        key = jrand.PRNGKey(42)       
        _x = jrand.normal(key, (3, 3))
        _y = jnp.eye(3)

        res = _x @ _y
            
        stx = SparseTensor([DenseDimension(0, 3, 0)], 
                        [DenseDimension(1, 3, 1)], _x)
        sty = SparseTensor([SparseDimension(0, 3, None, 1)], 
                        [SparseDimension(1, 3, None, 0)], None)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_simple_None_dense(self):        
        key = jrand.PRNGKey(42)  
        _x = jnp.eye(3)
        _y = jrand.normal(key, (3, 3))

        res = _x @ _y
            
        stx = SparseTensor([SparseDimension(0, 3, None, 1)], 
                        [SparseDimension(1, 3, None, 0)], None)
        sty = SparseTensor([DenseDimension(0, 3, 1)], 
                        [DenseDimension(1, 3, 0)], _y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    ## Tests for 3d tensors
    def test_3d_dense_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        y = jrand.normal(ykey, (5, 2, 2))
        res = jnp.einsum("ijk,klm->ijlm", x, y)
        
        stx = SparseTensor([DenseDimension(0, 3, 0), DenseDimension(1, 4, 1), ], 
                        [DenseDimension(2, 5, 2)], x)
        sty = SparseTensor([DenseDimension(0, 5, 0)], 
                        [DenseDimension(1, 2, 1), DenseDimension(2, 2, 2)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.allclose(res, stres.val))

    def test_3d_dense_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 4, 5))
        y = jrand.normal(ykey, (4, 5, 2))
        res = jnp.einsum("ijk,jkl->il", x, y)
        
        stx = SparseTensor([DenseDimension(0, 3, 0)], 
                        [DenseDimension(1, 4, 1), DenseDimension(2, 5, 2)], x)
        sty = SparseTensor([DenseDimension(0, 4, 0), DenseDimension(1, 5, 1)], 
                        [DenseDimension(2, 2, 2)], y)
        stres = stx * sty
        
        self.assertTrue(jnp.all(res == stres.val))

    def test_3d_sparse_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 5))
        d = jnp.eye(5)
        _x = jnp.einsum("ij,jk->ijk", x, d)
        
        y = jrand.normal(ykey, (5, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ij,jk->ikj", d, y)
        res = jnp.einsum("ijk,klm->ijlm", _x, _y)
                
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 5, 1, 2), ], 
                        [SparseDimension(2, 5, 1, 1)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 2)], 
                        [DenseDimension(1, 2, 1), SparseDimension(2, 5, 0, 0)], y)
        stres = stx * sty 
        
        iota = jnp.eye(5)
        
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_3d_2d_sparse_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        x = jrand.normal(xkey, (3, 5))
        d = jnp.eye(5)
        _x = jnp.einsum("ij,jk->ijk", x, d)
        
        y = jrand.normal(ykey, (5,))
        d = jnp.eye(5)
        _y = y*d
        res = jnp.einsum("ijk,kl->ijl", _x, _y)
            
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 5, 1, 2)], 
                        [SparseDimension(2, 5, 1, 1)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 1)], 
                        [SparseDimension(1, 5, 0, 0)], y)
        stres = stx * sty
        
        iota = jnp.eye(5)

        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_2d_3d_sparse_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (5,))
        d = jnp.eye(5)
        _x = x*d
        
        y = jrand.normal(ykey, (5, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ij,jk->ijk", d, y)
        res = jnp.einsum("ij,jkl->ikl", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 5, 0, 1)], 
                        [SparseDimension(1, 5, 0, 0)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 1)], 
                        [SparseDimension(1, 5, 0, 0), DenseDimension(2, 2, 1)], y)
        stres = stx * sty
        
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
    
    ### Tests for 4d tensors
    def test_4d_sparse_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,jl->ijkl", x, d)
        
        y = jrand.normal(ykey, (5, 4, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ijk,il->ijlk", y, d)
        
        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 4, 1, 3)], 
                        [DenseDimension(2, 5, 2), SparseDimension(3, 4, 1, 1)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 2), DenseDimension(1, 4, 1)], 
                        [SparseDimension(2, 5, 0, 0), DenseDimension(3, 2, 2)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
                
        print(stres.val, stres.val.shape, stres.val.sum())
        print(res, res.shape, res.sum())
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_4d_sparse_cross_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4, 3, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,il->ijkl", x, d)
        
        y = jrand.normal(ykey, (5, 4, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ijk,il->ijkl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 4, 0, 3), DenseDimension(1, 3, 1)], 
                        [DenseDimension(2, 5, 2), SparseDimension(3, 4, 0, 0)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 3), DenseDimension(1, 4, 1)], 
                        [DenseDimension(2, 2, 2), SparseDimension(3, 5, 0, 0)], y)
        stres = stx * sty
        
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_4d_softmax_sparse_single_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4, 3, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ijk,il->ijlk", x, d)
        
        y = jrand.normal(ykey, (4, 5, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ijk,jl->ijkl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 4, 0, 2), DenseDimension(1, 3, 1), ], 
                        [SparseDimension(2, 4, 0, 0), DenseDimension(3, 5, 2)], x)
        sty = SparseTensor([DenseDimension(0, 4, 0), SparseDimension(1, 5, 1, 3), ], 
                        [DenseDimension(2, 2, 2), SparseDimension(3, 5, 1, 1)], y)
        
        stres = stx * sty
        
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))

    def test_4d_softmax_sparse_single_contraction_with_Nones(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 5))
        d = jnp.eye(4)
        _x = jnp.einsum("ij,kl->ikjl", x, d)
        
        y = jrand.normal(ykey, (5, 4, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ijk,il->ijkl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 4, None, 3)], 
                        [DenseDimension(2, 5, 1), SparseDimension(3, 4, None, 1)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 3), DenseDimension(1, 4, 1)], 
                        [DenseDimension(2, 2, 2), SparseDimension(3, 5, 0, 0)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_4d_sparse_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 5))
        d = jnp.eye(15, 15)
        d = d.reshape(3, 5, 3, 5)
        _x = jnp.einsum("ij,ijkl->ijkl", x, d)
        
        y = jrand.normal(ykey, (3, 5))
        d = jnp.eye(15, 15)
        d = d.reshape(3, 5, 3, 5)
        _y = jnp.einsum("ij,ijkl->ijkl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, 0, 2), SparseDimension(1, 5, 1, 3)], 
                        [SparseDimension(2, 3, 0, 0), SparseDimension(3, 5, 1, 1)], x)
        sty = SparseTensor([SparseDimension(0, 3, 0, 2), SparseDimension(1, 5, 1, 3)], 
                        [SparseDimension(2, 3, 0, 0), SparseDimension(3, 5, 1, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(15)
                
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    
    def test_4d_dense_double_contraction(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4, 5, 6))
        y = jrand.normal(ykey, (5, 6, 2, 7))

        res = jnp.einsum("ijkl,klmn->ijmn", x, y)
            
        stx = SparseTensor([DenseDimension(0, 3, 0), DenseDimension(1, 4, 1)], 
                        [DenseDimension(2, 5, 2), DenseDimension(3, 6, 3)], x)
        sty = SparseTensor([DenseDimension(0, 5, 0), DenseDimension(1, 6, 1)], 
                        [DenseDimension(2, 2, 2), DenseDimension(3, 7, 3)], y)
        stres = stx * sty
                
        iota = jnp.eye(15)
                
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_sparse_single_contraction_with_two_Nones(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4, 5))
        d = jnp.eye(3)
        _x = jnp.einsum("ij,kl->kilj", x, d)
        
        y = jrand.normal(ykey, (3, 2))
        d = jnp.eye(5)
        _y = jnp.einsum("ij,kl->ikjl", y, d)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, None, 2), DenseDimension(1, 4, 0)], 
                        [SparseDimension(2, 3, None, 0), DenseDimension(3, 5, 1)], x)
        sty = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 5, None, 3)], 
                        [DenseDimension(2, 2, 1), SparseDimension(3, 5, None, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_sparse_dense_with_only_Nones(self): 
        key = jrand.PRNGKey(42)
        x = jrand.normal(key, (3, 5, 4))       
        d = jnp.eye(3)
        _x = jnp.einsum("ij,ikl->ikjl", d, x)
        
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _y = jnp.einsum("ij,kl->iklj", d1, d2)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, 0, 2), DenseDimension(1, 5, 1)], 
                        [SparseDimension(2, 3, 0, 0), DenseDimension(3, 4, 2)], x)
        sty = SparseTensor([SparseDimension(0, 3, None, 3), SparseDimension(1, 4, None, 2)], 
                        [SparseDimension(2, 4, None, 1), SparseDimension(3, 3, None, 0)], None)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_sparse_Nones(self): 
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 5, 4))       
        d = jnp.eye(3)
        _x = jnp.einsum("ij,ikl->ikjl", d, x)
        
        y = jrand.normal(ykey, (3,))
        d1 = jnp.eye(3)
        _y = d1*y
        d2 = jnp.eye(4)
        _y = jnp.einsum("ij,kl->iklj", _y, d2)

        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, 0, 2), DenseDimension(1, 5, 1)], 
                        [SparseDimension(2, 3, 0, 0), DenseDimension(3, 4, 2)], x)
        sty = SparseTensor([SparseDimension(0, 3, 0, 3), SparseDimension(1, 4, None, 2)], 
                        [SparseDimension(2, 4, None, 1), SparseDimension(3, 3, 0, 0)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_only_Nones(self):        
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _x = jnp.einsum("ij,kl->ikjl", d1, d2)
        
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _y = jnp.einsum("ij,kl->iklj", d1, d2)

        print(_x.shape, _y.shape)
        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, None, 2), SparseDimension(1, 4, None, 3)], 
                        [SparseDimension(2, 3, None, 0), SparseDimension(3, 4, None, 1)], None)
        sty = SparseTensor([SparseDimension(0, 3, None, 3), SparseDimension(1, 4, None, 2)], 
                        [SparseDimension(2, 4, None, 1), SparseDimension(3, 3, None, 0)], None)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_double_sparse_single_sparse_Nones(self):   
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4))    
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _x = jnp.einsum("ij,ik,jl->ijlk", x, d1, d2)
        
        y = jrand.normal(ykey, (3, 5))
        d = jnp.eye(4)
        _y = jnp.einsum("ij,kl->ikjl", d, y)

        print(_x.shape, _y.shape)
        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, 0, 3), SparseDimension(1, 4, 1, 2)], 
                        [SparseDimension(2, 4, 1, 1), SparseDimension(3, 3, 0, 0)], x)
        sty = SparseTensor([SparseDimension(0, 4, None, 2), DenseDimension(1, 3, 0)], 
                        [SparseDimension(2, 4, None, 0), DenseDimension(3, 5, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
        
    def test_4d_so_tired_of_it(self):   
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4))    
        d1 = jnp.eye(3)
        d2 = jnp.eye(4)
        _x = jnp.einsum("ij,ik,jl->ijlk", x, d1, d2)
        
        y = jrand.normal(ykey, (4, 5))
        d = jnp.eye(3)
        _y = jnp.einsum("ij,kl->ikjl", y, d)

        print(_x.shape, _y.shape)
        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([SparseDimension(0, 3, 0, 3), SparseDimension(1, 4, 1, 2)], 
                        [SparseDimension(2, 4, 1, 1), SparseDimension(3, 3, 0, 0)], x)
        sty = SparseTensor([DenseDimension(0, 4, 0), SparseDimension(1, 3, None, 3)], 
                        [DenseDimension(2, 5, 1), SparseDimension(3, 3, None, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    def test_4d_dense_None_None_dense(self):   
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (3, 4))    
        d = jnp.eye(5)
        _x = jnp.einsum("ij,kl->ikjl", x, d)
        
        y = jrand.normal(ykey, (5, 2))
        d = jnp.eye(4)
        _y = jnp.einsum("ij,kl->ikjl", d, y)

        print(_x.shape, _y.shape)
        res = jnp.einsum("ijkl,klmn->ijmn", _x, _y)
            
        stx = SparseTensor([DenseDimension(0, 3, 0), SparseDimension(1, 5, None, 3)], 
                        [DenseDimension(2, 4, 1), SparseDimension(3, 5, None, 1)], x)
        sty = SparseTensor([SparseDimension(0, 4, None, 2), DenseDimension(1, 5, 0)], 
                        [SparseDimension(2, 4, None, 0), DenseDimension(3, 2, 1)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))
        
    ### Replication tests
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
        
    def test_3d_4d_sparse(self):
        key = jrand.PRNGKey(42)
        xkey, ykey = jrand.split(key, 2)
        
        x = jrand.normal(xkey, (4, 5))
        _x = jnp.einsum("ij,ik->ijk", x, jnp.eye(4))
        
        y = jrand.normal(ykey, (5, 4, 3))
        _y = jnp.einsum("ijk,il->ijlk", y, jnp.eye(5))

        print(_x.shape, _y.shape)
        res = jnp.einsum("ijk,jklm->ilm", _x, _y)
        
        stx = SparseTensor([SparseDimension(0, 4, 0, 2)], 
                        [DenseDimension(1, 5, 1), SparseDimension(2, 4, 0, 0)], x)
        sty = SparseTensor([SparseDimension(0, 5, 0, 2), DenseDimension(1, 4, 1)], 
                        [SparseDimension(2, 5, 0, 0), DenseDimension(3, 3, 2)], y)
        stres = stx * sty
                
        iota = jnp.eye(5)
            
        self.assertTrue(jnp.allclose(res, stres.dense(iota)))


if __name__ == '__main__':
    unittest.main()

