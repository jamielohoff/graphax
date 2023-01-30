from typing import Callable, Tuple
from functools import partial
import copy

import jax
import jax.lax as lax

import chex

from .graph import GraphState, is_bipartite
from .elimination import eliminate, front_eliminate, back_eliminate


def vert_elim(info: chex.Array, gs: GraphState, vertex: int):
    return eliminate(gs, vertex, info)


def front_elim(info: chex.Array, gs: GraphState, edge: Tuple[int, int]):
    return front_eliminate(gs, edge, info)


def back_elim(info: chex.Array, gs: GraphState, edge: Tuple[int, int]):
    return back_eliminate(gs, edge, info)


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
        
        new_gs, nops = self.vertex_eliminate(gs, vertex)

        reward = -nops
        
        terminated = lax.cond(is_bipartite(new_gs),
                                    lambda: True,
                                    lambda: False)
    
        return new_gs, reward, terminated
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> GraphState:
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
        self.num_inputs = gs.get_info()[0]
        self.front_eliminate = partial(front_elim, gs.get_info())
        self.back_eliminate = partial(back_elim, gs.get_info())
    
    # Remove JIT compilation here
    @partial(jax.jit, static_argnums=(0,))
    def step(self,
            gs: GraphState,
            action: chex.Array) -> Tuple[GraphState, float, bool]:
        i, j, mode = action

        i = i - self.num_inputs + 1
        j += 1

        new_gs, nops = lax.cond(mode == 0,
                            lambda g: self.front_eliminate(g, (i,j)),
                            lambda g: self.back_eliminate(g, (i,j)),
                            gs)
                
        reward = -nops
    
        terminated = lax.cond(is_bipartite(new_gs),
                            lambda: True,
                            lambda: False)
        return new_gs, reward, terminated
    
    # remove jit compilation here!
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> GraphState:
        return copy.deepcopy(self.gs)

