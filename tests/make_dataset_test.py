from graphax.dataset import Graph2File
from graphax.dataset import read, get_prompt_list

API_KEY = "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"
PROMPT_LIST = get_prompt_list("./tests/prompt_list.txt")    

gen = Graph2File(API_KEY, "./tests", PROMPT_LIST)

gen.generate()

codes, graphs, infos = read("comp_graph_examples-0.hdf5", [i for i in range(20)])

print(graphs)

