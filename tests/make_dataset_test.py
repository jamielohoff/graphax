from graphax.dataset import Graph2File
from graphax.dataset import read

API_KEY = "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"
MESSAGE = "Generate an arbitrary JAX function with 4 inputs and \
            4 outputs and a maximum of 15 operations. Name the \
            function 'f' and only show the source code without \
            jit and description."
MAKE_JAXPR = "jaxpr = jax.make_jaxpr(f)(1., 1., 1., 1.)"
PROMPT_LIST = [(MESSAGE, MAKE_JAXPR)]          

gen = Graph2File(API_KEY, "./", PROMPT_LIST)

gen.generate()

codes, graphs, infos = read("comp_graph_examples-0.hdf5", [0,1,2,3,4])

print(graphs)

