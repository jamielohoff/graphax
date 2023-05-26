from typing import Sequence, Tuple
import openai

import jax
import jax.numpy as jnp

import chex

from .utils import check_graph_shape
from ..core import GraphInfo, make_graph_info
from ..interpreter.from_jaxpr import make_graph
from ..transforms import safe_preeliminations_gpu, compress_graph, embed


# TODO refactor code such that we do no longer need the global variable
jaxpr = ""

class ComputationalGraphSampler:
    """
    Class that implements a sampling function using ChatGPT to create realistic
    examples of computational graphs.
    
    Returns Jaxpr objects or string defining the function
    """
    api_key: str
    default_message: str
    default_make_jaxpr: str
    max_info: GraphInfo
    
    def __init__(self, 
                api_key: str, 
                default_message: str,
                default_make_jaxpr: str,
                max_info: Tuple[int, int, int] = make_graph_info([10, 30, 5])) -> None:
        self.api_key = api_key
        self.default_message = default_message
        self.default_make_jaxpr = default_make_jaxpr
        self.max_info = max_info
    
    def sample(self, 
               num_samples: int = 1, 
               message: str = None,
               make_jaxpr: str = None,
               **kwargs) -> Sequence[tuple[str, chex.Array, GraphInfo]]:
        openai.api_key = self.api_key
        message = self.default_message if message is None else message
        make_jaxpr = "\n"+self.default_make_jaxpr if make_jaxpr is None else "\n"+make_jaxpr

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
            
            if len(clean_lines) == 0:
                continue

            function = "\n".join(clean_lines)
            function += make_jaxpr
            try:
                exec(function, globals())
                edges, info = make_graph(jaxpr)
                edges, info = safe_preeliminations_gpu(edges, info)
                edges, info = compress_graph(edges, info)
                edges, _, vertices, attn_mask = embed(edges, info, self.max_info)
                print(info)
                if check_graph_shape(info, self.max_info):
                    samples.append((function, edges, info, vertices, attn_mask))
            except Exception as e:
                print(e)
                continue
        
        return samples

