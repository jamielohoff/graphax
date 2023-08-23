from typing import Sequence, Tuple
import openai

import jax
import jax.random as jrand

from chex import Array, PRNGKey

from .sampler import ComputationalGraphSampler
from .utils import check_graph_shape
from ..interpreter.from_jaxpr import make_graph
from ..transforms import safe_preeliminations, compress, embed


# TODO refactor code such that we do no longer need the global variable
jaxpr = ""

class LLMSampler(ComputationalGraphSampler):
    """
    Class that implements a sampling function using ChatGPT to create realistic
    examples of computational graphs.
    
    Returns jaxpr objects or string defining the function
    """
    api_key: str
    prompt_list: Sequence[Tuple[str, str]]
    sleep_timer: int
    
    def __init__(self, 
                api_key: str, 
                prompt_list: Sequence[Tuple[str, str]],
                *args,
                sleep_timer: int = 12,
                **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        
        self.api_key = api_key
        self.prompt_list = prompt_list
        self.sleep_timer = sleep_timer
    
    def sample(self, 
               num_samples: int = 1, 
               key: PRNGKey = None,
               **kwargs) -> Sequence[tuple[str, Array]]:
        openai.api_key = self.api_key
        idx = jrand.randint(key, (), 0, len(self.prompt_list)-1)
        message, make_jaxpr = self.prompt_list[idx]
        make_jaxpr = "\n"+make_jaxpr

        # Define prompt
        messages = [{"role": "user", "content": message}]
        samples = []

        # Use the API to generate a response
        responses = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=messages,
                                                n=num_samples,
                                                stop=None, 
                                                **kwargs)
        
        for response in responses.choices:
            function = response.message.content
            # print(function)
            lines = function.split("\n")
            clean_lines = []
            indicator = False
            for line in lines:
                if "import" in line:
                    indicator = True
                if indicator:
                    clean_lines.append(line)
                if "return" in line:
                    indicator = False
            
            if len(clean_lines) == 0: continue

            function = "\n".join(clean_lines)
            function += make_jaxpr
            try:
                exec(function, globals())
                edges = make_graph(jaxpr)
                edges = safe_preeliminations(edges)
                edges = compress(edges)
                edges = embed(edges, self.max_info)
                
                info = edges.at[0, 0, 0:3].get()
                if check_graph_shape(info, self.max_info) and info[1] > self.min_num_intermediates:
                    samples.append((function, edges))
            except Exception as e:
                print(e)
                continue
        
        return samples

