from typing import Sequence
import openai

import jax
import jax.numpy as jnp

from graphax.interpreter.from_jaxpr import make_graph

API_KEY = "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"
MESSAGE = """Generate an arbitrary JAX function with 4 inputs and 
            4 outputs and a maximum of 10 operations. Name the 
            function 'f' and only show the source code without 
            jit and description."""

# TODO refactor code such that we do no longer need the global variable
jaxpr = ""

class ComputationalGraphSampler:
    """
    Class that implements a sampling function using ChatGPT to create realistic
    samples of computational graphs.
    """
    api_key: str
    default_message: str
    
    def __init__(self, api_key: str, default_message: str) -> None:
        self.api_key = api_key
        self.default_message = default_message
    
    def sample(self, 
               num_samples: int = 1, 
               message: str = None,
               **kwargs) -> Sequence[str]:
        openai.api_key = self.api_key
        message = self.default_message if message is None else message

        # Define prompt
        messages = [{"role": "user",
                    "content": message}]

        # Use the API to generate a response
        responses = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                messages=messages,
                                                n=num_samples,
                                                stop=None, 
                                                **kwargs)
        
        make_jaxpr = "\njaxpr = jax.make_jaxpr(f)(1., 1., 1., 1.)"
        for response in responses.choices:
            # Print the generated response
            function = response.message.content
            lines = function.split("\n")
            function = "\n".join(lines[1:-1])
            function += make_jaxpr
            print(function)
            exec(function, globals())
            print("Output:", make_graph(jaxpr))
        

gen = ComputationalGraphSampler(api_key=API_KEY, default_message=MESSAGE)
gen.sample(temperature=0.75, max_tokens=300)



