import numpy as np

from scipy.sparse import lil_matrix
from .graph import GraphState, add_edge

def construct_Helmholtz():
    n = 4 + 11 + 4
    edges = lil_matrix((n, n), dtype=np.float32)
    state = np.zeros((11,))
    gs = GraphState(edges, state, 4, 11, 4)
    
    gs = add_edge(gs, (0,4), .5)
    gs = add_edge(gs, (0,7), .5)
    gs = add_edge(gs, (0,15), .5)
    
    gs = add_edge(gs, (1,4), .5)
    gs = add_edge(gs, (1,8), .5)
    gs = add_edge(gs, (1,16), .5)
    
    gs = add_edge(gs, (2,4), .5)
    gs = add_edge(gs, (2,9), .5)
    gs = add_edge(gs, (2,17), .5)
    
    gs = add_edge(gs, (3,4), .5)
    gs = add_edge(gs, (3,10), .5)
    gs = add_edge(gs, (3,18), .5)
    
    gs = add_edge(gs, (4,5), .5)
    
    gs = add_edge(gs, (5,6), .5)
    
    gs = add_edge(gs, (6,7), .5)
    gs = add_edge(gs, (6,8), .5)
    gs = add_edge(gs, (6,9), .5)
    gs = add_edge(gs, (6,10), .5)
    
    gs = add_edge(gs, (7,11), .5)
    gs = add_edge(gs, (8,12), .5)
    gs = add_edge(gs, (9,13), .5)
    gs = add_edge(gs, (10,14), .5)
    
    gs = add_edge(gs, (11,15), .5)
    gs = add_edge(gs, (12,16), .5)
    gs = add_edge(gs, (13,17), .5)
    gs = add_edge(gs, (14,18), .5)
    return gs

