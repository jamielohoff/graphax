import numpy as np
import copy

from .graph import GraphState, is_bipartite
from .elimination import eliminate

class VertexGame:
    """
    static environment for the game 
    
    
    game always has finite termination range!
    """       
    @staticmethod
    def step(gs: GraphState, action: np.ndarray):
        vertex = action
            
        gs, nmults, nadds = eliminate(gs, vertex + gs.ninputs)

        reward = -nmults
        gs.state[vertex] = 1.
    
        if is_bipartite(gs):
            terminated = True
        else:
            terminated = False

        return gs, reward, terminated
    
    @staticmethod
    def reset(backup_gs: GraphState):
        return copy.deepcopy(backup_gs)
        
