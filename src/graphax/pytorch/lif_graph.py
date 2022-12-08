import numpy as np
from scipy.sparse import lil_matrix

from .graph import GraphState, add_edge


def construct_LIF():
    n = 6 + 9 + 3
    edges = lil_matrix((n, n), dtype=np.float32)
    state = np.zeros((9,))
    gs = GraphState(edges, state, 6, 9, 3)
    
    gs = add_edge(gs, (0,6), .5)
    
    gs = add_edge(gs, (1,6), .5)
    gs = add_edge(gs, (1,7), .5)
    
    gs = add_edge(gs, (2,8), .5)
    gs = add_edge(gs, (2,10), .5)
    
    gs = add_edge(gs, (3,8), .5)
    gs = add_edge(gs, (3,9), .5)
    
    gs = add_edge(gs, (4,11), .5)
    
    gs = add_edge(gs, (5,14), .5)
    
    gs = add_edge(gs, (6,12), .5)
    
    gs = add_edge(gs, (7,10), .5)
    
    gs = add_edge(gs, (8,13), .5)
    
    gs = add_edge(gs, (9,11), .5)
    
    gs = add_edge(gs, (10,12), .5)
    
    gs = add_edge(gs, (11,13), .5)
    
    gs = add_edge(gs, (12,14), .5)
    gs = add_edge(gs, (12,16), .5)
    
    gs = add_edge(gs, (13,17), .5)
    
    gs = add_edge(gs, (14,15), .5)
    gs = add_edge(gs, (14,16), .5) # gated reset

    return gs

