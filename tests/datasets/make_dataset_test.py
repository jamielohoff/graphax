import jax
import jax.random as jrand

from graphax.dataset import Graph2File, read, get_prompt_list, LLMSampler, RandomSampler

API_KEY = "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"
# PROMPT_LIST = get_prompt_list("./prompt_list.txt")   
key = jrand.PRNGKey(42) 

sampler = RandomSampler(max_info=[5,10,5], min_num_intermediates=12)

gen = Graph2File(sampler, "./", sampler_batchsize=64, num_samples=128, samples_per_file=64, max_info=[5, 10, 5])

gen.generate(key=key)

codes, graphs = read("comp_graph_examples-0.hdf5", [i for i in range(63)])

print(graphs)

