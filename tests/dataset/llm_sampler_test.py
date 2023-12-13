import jax
import jax.numpy as jnp
import jax.random as jrand

import graphax as gx
from graphax.dataset import LLMSampler, densify


API_KEY = "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"
CARO_KEY = "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"
MESSAGE = """Generate an arbitrary JAX function with 5 imputs and 
            3 outputs and a minimum of intermediate 50 operations. 
            Primarily use functions with more than one input.
            Name the function 'f' and only show the source code without 
            jit and description."""
MAKE_JAXPR = "jaxpr = jax.make_jaxpr(f)(1., 1., 1., 1., 1.)"
            

gen = LLMSampler(api_key=CARO_KEY, 
                prompt_list=[(MESSAGE, MAKE_JAXPR)],
                min_num_intermediates=10)
key = jrand.PRNGKey(42)
samples = gen.sample(num_samples=1,
                    key=key, 
                    temperature=0.75, 
                    max_tokens=2500)

code, header, sparse_edges = samples[0]
print(code)
edges = densify(header, sparse_edges)
edges = jnp.array(edges)
print(edges)
print(gx.forward(edges))
print(gx.reverse(edges))

