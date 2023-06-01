import jax
import jax.random as jrand

from graphax.dataset import Graph2File, read, get_prompt_list, LLMSampler, RandomSampler

API_KEY = "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"
PROMPT_LIST = get_prompt_list("./prompt_list.txt")   
key = jrand.PRNGKey(42) 

sampler = RandomSampler(16, min_num_intermediates=12)

gen = Graph2File(sampler, "./", sampler_batchsize=32)

gen.generate(key=key, minval=0.05, maxval=0.5)

codes, graphs, infos, vertices, attn_masks = read("comp_graph_examples-0.hdf5", [i for i in range(99)])

print(graphs)

