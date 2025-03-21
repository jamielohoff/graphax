import time
import unittest
from functools import partial

import jax
import jax.nn as jnn
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jrand
from jax.tree_util import tree_map

from graphax import jacve, tree_allclose
from graphax.examples.neuromorphic import ADALIF_SNN


class SNNTest(unittest.TestCase): 
    def test_snn_step(self):
        key = jrand.PRNGKey(42)
        
        alpha = .95
        beta = .85
        rho = .9
        thresh = 1.
        
        S_in = jnp.ones((160,), dtype=jnp.float32)
        S_target = jnp.ones((10,), dtype=jnp.float32)
        U1 = jnp.zeros(128, dtype=jnp.float32)
        U2 = jnp.zeros(64, dtype=jnp.float32)
        U3 = jnp.zeros(10, dtype=jnp.float32)
        
        a1 = jnp.zeros(128, dtype=jnp.float32)
        a2 = jnp.zeros(64, dtype=jnp.float32)
        a3 = jnp.zeros(10, dtype=jnp.float32)
        
        w1key, w2key, w3key, key = jrand.split(key, 4)
        W1 = jrand.normal(w1key, (128, 160))
        W2 = jrand.normal(w2key, (64, 128))
        W3 = jrand.normal(w3key, (10, 64))
        
        print(jax.make_jaxpr(ADALIF_SNN)(S_in, S_target, U1, U2, U3, 
                                        a1, a2, a3, W1, W2, W3, 
                                        alpha, beta, rho, thresh))
        
        argnums = list(range(2, 15))
        # print(jax.make_jaxpr(jacve(ADALIF_SNN, order="rev", argnums=argnums))(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh))
        deriv_fn = jax.jit(jacve(ADALIF_SNN, order="rev", argnums=argnums))
        veres = deriv_fn(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh)
        
        jax_deriv_fn = jax.jit(jax.jacrev(ADALIF_SNN, argnums=argnums))
        revres = jax_deriv_fn(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh)
                                        
        self.assertTrue(tree_allclose(veres, revres))
        
    def test_snn_simple_rollout(self):
        key = jrand.PRNGKey(42)
        seq_len = 25
        argnums = range(2, 15)
        deriv_fn = jacve(ADALIF_SNN, order="rev", argnums=argnums)
        
        def recursive_eprop_relation():
            pass
        
        def get_eligibility_trace():
            pass
        
        def rollout(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh):
            loss = 0.
            for i in range(seq_len):
                
                l, U1, U2, U3, a1, a2, a3 = deriv_fn(S_in[i], S_target[i], U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh)
                loss += l
            return loss
        
        alpha = .95
        beta = .85
        rho = .9
        thresh = 1.
        
        S_in = jnp.ones((seq_len, 160), dtype=jnp.float32)
        S_target = jnp.ones((seq_len, 10,), dtype=jnp.float32)
        U1 = jnp.zeros(128, dtype=jnp.float32)
        U2 = jnp.zeros(64, dtype=jnp.float32)
        U3 = jnp.zeros(10, dtype=jnp.float32)
        
        a1 = jnp.zeros(128, dtype=jnp.float32)
        a2 = jnp.zeros(64, dtype=jnp.float32)
        a3 = jnp.zeros(10, dtype=jnp.float32)
        
        w1key, w2key, w3key, key = jrand.split(key, 4)
        W1 = jrand.normal(w1key, (128, 160))
        W2 = jrand.normal(w2key, (64, 128))
        W3 = jrand.normal(w3key, (10, 64))
        
        print(jax.make_jaxpr(ADALIF_SNN)(S_in, S_target, U1, U2, U3, 
                                        a1, a2, a3, W1, W2, W3, 
                                        alpha, beta, rho, thresh))
        
        argnums = range(2, 15)
        print(jax.make_jaxpr(jacve(ADALIF_SNN, order="rev", argnums=argnums))(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh))
        deriv_fn = jacve(ADALIF_SNN, order="rev", argnums=argnums)
        veres = deriv_fn(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh)
        
        jax_deriv_fn = jax.jit(jax.jacrev(ADALIF_SNN, argnums=argnums))
        revres = jax_deriv_fn(S_in, S_target, U1, U2, U3, a1, a2, a3, W1, W2, W3, alpha, beta, rho, thresh)
                                        
        self.assertTrue(tree_allclose(veres, revres))

if __name__ == '__main__':
    unittest.main()   
        
    