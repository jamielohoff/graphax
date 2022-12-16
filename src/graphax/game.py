from typing import Callable, Tuple
from functools import partial
import copy

import jax
import jax.lax as lax

from .graph import GraphState, is_bipartite
from .elimination import eliminate, front_eliminate, back_eliminate
from jaxtyping import Array


def vert_elim(info: Array, gs: GraphState, vertex: int):
    return eliminate(gs, vertex, info)


def front_elim(info: Array, gs: GraphState, edge: Tuple[int, int]):
    return front_eliminate(gs, edge, info)


def back_elim(info: Array, gs: GraphState, edge: Tuple[int, int]):
    return back_eliminate(gs, edge, info)


# TODO implement edge game
class VertexGame:
    """
    OpenAI-like gymnax environment for the game
    
    The game always has finite termination range!
    """
    gs: GraphState
    vertex_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        self.vertex_eliminate = partial(vert_elim, gs.get_info())
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: int) -> Tuple[GraphState, float, bool]:        
        vertex = action + 1
        
        gs, nops = self.vertex_eliminate(gs, vertex)

        reward = -nops
        
        gs, terminated = lax.cond(is_bipartite(gs),
                    lambda g: (self.reset(), True),
                    lambda g: (g, False),
                    gs)
    
        return gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> GraphState:
        return copy.deepcopy(self.gs)


class EdgeGame:
    """
    OpenAI-like gymnax environment for the game
    
    The game always has finite termination range!
    """
    gs: GraphState
    ninputs: int
    front_eliminate: Callable
    back_eliminate: Callable
    
    def __init__(self, gs: GraphState) -> None:
        super().__init__()
        self.gs = copy.deepcopy(gs)
        self.ninputs = gs.get_info()[0]
        self.front_eliminate = partial(front_elim, gs.get_info())
        self.back_eliminate = partial(back_elim, gs.get_info())
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: Array) -> Tuple[GraphState, float, bool]:
        i, j, mode = action

        i = i - self.ninputs + 1
        j += 1

        gs, nops = lax.cond(mode == 0,
                            lambda g: self.front_eliminate(g, (i,j)),
                            lambda g: self.back_eliminate(g, (i,j)),
                            gs)
        
        reward = -nops
    
        gs, terminated = lax.cond(is_bipartite(gs),
                            lambda g: (self.reset(), True),
                            lambda g: (g, False),
                            gs)
        return gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> GraphState:
        return copy.deepcopy(self.gs)

