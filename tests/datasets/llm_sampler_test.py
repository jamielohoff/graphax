import graphax as gx
from graphax.dataset import ComputationalGraphSampler
from graphax.transforms import safe_preeliminations, compress


API_KEY = "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"
MESSAGE = """Generate an arbitrary JAX function with a input vector of size 4 and 
            4 outputs and a maximum of 10 operations. 
            Include a jnp.sum operation over multiple variables. 
            Name the function 'f' and only show the source code without 
            jit and description."""
MAKE_JAXPR = "jaxpr = jax.make_jaxpr(f)(jnp.ones(4))"
            

gen = ComputationalGraphSampler(api_key=API_KEY, 
                                default_message=MESSAGE,
                                default_make_jaxpr=MAKE_JAXPR)
fn_list, graph_list, info_list = gen.sample(temperature=0.75, max_tokens=300)
edges, info = graph_list[0]
print(graph_list)
edges, info = gx.safe_preeliminations(edges)
edges, info = gx.compress(edges)
print(gx.forward_gpu(edges))
print(gx.reverse_gpu(edges))

