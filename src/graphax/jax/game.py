from typing import Callable, Tuple
from functools import partial
import copy

import chex
import jax
import jax.lax as lax

from .graph import GraphState, is_bipartite
from .elimination import eliminate


def vert_elim(nedges: int, ninputs: int, gs: GraphState, vertex: int):
    return eliminate(gs, vertex, nedges, ninputs)


# TODO implement edge game
class VertexGame:
    """
    OpenAI-like gymnax environment for the game
    
    game always has finite termination range!
    """
    gs: GraphState
    ninputs: int
    nintermediates: int
    noutputs: int
    vertex_eliminate: Callable
    
    def __init__(self, 
                gs: GraphState, 
                nedges: int, 
                ninputs: int, 
                nintermediates: int, 
                noutputs: int) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        
        self.ninputs = ninputs
        self.nintermediates = nintermediates
        self.noutputs = noutputs
        
        self.vertex_eliminate = partial(vert_elim, nedges, ninputs)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: int) -> Tuple[GraphState, float, bool]:
        vertex = action
        
        gs, nops = self.vertex_eliminate(gs, vertex + self.ninputs)

        reward = -nops
    
        gs, terminated = lax.cond(is_bipartite(gs, self.nintermediates),
                            lambda g: (self.reset(), True),
                            lambda g: (g, False),
                            gs)
        return gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> GraphState:
        return copy.deepcopy(self.gs)

