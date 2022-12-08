from typing import Tuple, NamedTuple

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

class GraphState(NamedTuple):
    """The state of the graph is the connectivity of the edges represented
    as a sparse matrix plus the already eliminated vertices

    Args:
        NamedTuple (_type_): _description_
    """
    edges: np.ndarray
    state: np.ndarray
    
    ninputs: int
    nintermediates: int
    noutputs: int


def add_edge(gs: GraphState, pos: Tuple[int, int], val: float) -> GraphState:
    """
    TODO add docstring
    """
    gs.edges[pos[0], pos[1]] = val
    return gs      


def is_bipartite(gs: GraphState) -> bool:
    """TODO docstring

    Returns:
        _type_: _description_
    """
    return gs.state.sum() == gs.nintermediates
  
    
# TODO implement proper drawing function
def draw(gs: GraphState, fname: str) -> None:
    nxg = nx.DiGraph()
    in_nodes_connected = []
    for i, j in zip(*gs.edges.nonzero()):
        nxg.add_edge(i, j)
        if i < gs.ninputs and j < gs.ninputs+gs.nintermediates:
            in_nodes_connected.append(j)
                    
    in_nodes = list(range(gs.ninputs))
    out_nodes = list(range(gs.ninputs+gs.nintermediates, gs.ninputs+gs.nintermediates+gs.noutputs))
    
    for node in nxg.nodes.items():
        if node[0] in in_nodes:
            node[1]["subset"] = 0
        elif node[0] in in_nodes_connected:
            node[1]["subset"] = 1
        elif node[0] in out_nodes:
            node[1]["subset"] = 3
        else:
            node[1]["subset"] = 2
    pos = nx.multipartite_layout(nxg)
    plt.figure()
    nx.draw(nxg, pos=pos)
    plt.savefig(fname)

