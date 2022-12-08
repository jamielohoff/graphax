import numpy as np

from scipy.sparse import lil_matrix
from .graph import GraphState, add_edge


def construct_layered_graph(ninputs, nintermediates, noutputs, low=1, high=4, intermediate_split=2):
    n = ninputs + nintermediates + noutputs
    edges = lil_matrix((n, n), dtype=np.float32)
    state = np.zeros((nintermediates,))
    gs = GraphState(edges, 
                    state,
                    ninputs,
                    nintermediates,
                    noutputs)
    
    layer = ninputs
    next_layer = ninputs + nintermediates//intermediate_split
    
    for vertex in range(n-noutputs):
        if vertex > layer:
            layer += nintermediates//intermediate_split
            next_layer += nintermediates//intermediate_split
        for _ in range(np.random.randint(low=low, high=high)):
            other_vertex = np.random.randint(low=layer+1, high=min(next_layer, n))
            gs = add_edge(gs, (vertex, other_vertex), np.random.uniform())
    return gs


def construct_random_graph(ninputs, nintermediates, noutputs, low=1, high=4):
    
    n = ninputs + nintermediates + noutputs
    edges = lil_matrix((n, n), dtype=np.float32)
    state = np.zeros((nintermediates,))
    gs = GraphState(edges, 
                    state,
                    ninputs,
                    nintermediates,
                    noutputs)
    
    for vertex in range(n-noutputs):
        for _ in range(np.random.randint(low=low, high=high)):
            other_vertex = np.random.randint(low=vertex+1, high=n)
            gs = add_edge(gs, (vertex, other_vertex), np.random.uniform())
    return gs

