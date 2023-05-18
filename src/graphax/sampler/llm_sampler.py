from typing import Sequence
import openai

import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph

# TODO refactor code such that we do no longer need the global variable
jaxpr = ""

class ComputationalGraphSampler:
    """
    Class that implements a sampling function using ChatGPT to create realistic
    samples of computational graphs.
    
    Returns Jaxpr objects or string defining the function
    """
    api_key: str
    default_message: str
    default_make_jaxpr: str
    
    def __init__(self, 
                api_key: str, 
                default_message: str,
                default_make_jaxpr: str) -> None:
        self.api_key = api_key
        self.default_message = default_message
        self.default_make_jaxpr = default_make_jaxpr
    
    def sample(self, 
               num_samples: int = 1, 
               message: str = None,
               **kwargs) -> Sequence[str]:
        openai.api_key = self.api_key
        message = self.default_message if message is None else message

        # Define prompt
        messages = [{"role": "user",
                    "content": message}]
        
        # List to store the strings that define the function
        fn_list = []
        graph_list = []

        # Use the API to generate a response
        responses = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=messages,
                                                n=num_samples,
                                                stop=None, 
                                                **kwargs)
        
        make_jaxpr = "\n" + self.default_make_jaxpr
        for response in responses.choices:
            # Print the generated response
            function = response.message.content
            lines = function.split("\n")
            function = "\n".join(lines[1:-1])
            function += make_jaxpr
            fn_list.append(function)
            exec(function, globals())
            graph_list.append(make_graph(jaxpr))
        return fn_list, graph_list

