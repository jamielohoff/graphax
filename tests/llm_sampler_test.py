from graphax.sampler.llm_sampler import ComputationalGraphSampler

API_KEY = "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"
MESSAGE = """Generate an arbitrary JAX function with 4 inputs and 
            4 outputs and a maximum of 10 operations. Name the 
            function 'f' and only show the source code without 
            jit and description."""
MAKE_JAXPR = "jaxpr = jax.make_jaxpr(f)(1., 1., 1., 1.)"
            

gen = ComputationalGraphSampler(api_key=API_KEY, 
                                default_message=MESSAGE,
                                default_make_jaxpr=MAKE_JAXPR)
fn_list, graph_list = gen.sample(temperature=0.75, max_tokens=300)
print(fn_list)
print(graph_list)

            