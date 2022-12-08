from typing import Tuple

from .graph import GraphState

def front_eliminate(gs: GraphState, 
                    edge: Tuple[int, int]) -> Tuple[GraphState, int, int]:
    """TODO add docstring

    Args:
        edge (Tuple[int]): _description_
    """
    nmults, nadds = 0, 0
    edge_val = gs.edges[edge[0], edge[1]]
    gs.edges[edge[0], edge[1]] = 0.
    for i, j in zip(*gs.edges.nonzero()):
        val = gs.edges[i, j]
        if i == edge[1]:
            gs.edges[edge[0], j] += val*edge_val
            nmults += 1
            nadds += 1
    return gs, nmults, nadds

  
def back_eliminate(gs: GraphState, 
                    edge: Tuple[int, int]) -> Tuple[GraphState, int, int]:
    """TODO add docstring

    Args:
        edge (Tuple[int]): _description_
    """
    nmults, nadds = 0, 0
    edge_val = gs.edges[edge[0], edge[1]]
    gs.edges[edge[0], edge[1]] = 0.
    for i, j in zip(*gs.edges.nonzero()):
        val = gs.edges[i, j]
        if j == edge[0]:
            gs.edges[i, edge[1]] += val*edge_val
            nmults += 1
            nadds += 1
    return gs, nmults, nadds
    

def eliminate(gs: GraphState, 
            vertex: int) -> Tuple[GraphState, int, int]:
    """TODO add docstring

    Args:
        vertex (int): _description_
    """
    nmults, nadds = 0, 0
    for i, j in zip(*gs.edges.nonzero()):
        mults, adds = 0, 0
        # front-eliminiation of ingoing edges
        if j == vertex:
            gs, mults, adds = front_eliminate(gs, (i,j))
        # back-elimination of outgoing edges
        elif i == vertex:
            gs, mults, adds = back_eliminate(gs, (i,j))
        else:
            continue
            
        nmults += mults
        nadds += adds
    gs.state[vertex - gs.ninputs] = 1.
    return gs, nmults, nadds


def forward(gs: GraphState) -> Tuple[GraphState, int, int]:
    """TODO docstring

    Returns:
        _type_: _description_
    """
    nmults, nadds = 0, 0
    for i in range(gs.nintermediates):
        gs, mults, adds = eliminate(gs, i + gs.ninputs)
        nmults += mults
        nadds += adds
    return gs, nmults, nadds


def reverse(gs: GraphState) -> Tuple[GraphState, int, int]:
    """TODO docstring

    Returns:
        _type_: _description_
    """
    nmults, nadds = 0, 0
    for i in range(gs.nintermediates)[::-1]:
        gs, mults, adds = eliminate(gs, i + gs.ninputs)
        nmults += mults
        nadds += adds
    return gs, nmults, nadds

