from typing import Tuple

import chex

from ..core import GraphInfo, make_graph_info, make_empty_edges, add_edge


def make_LIF() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([6, 9, 3])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-5,1), info)
    
    edges = add_edge(edges, (-4, 1), info)
    edges = add_edge(edges, (-4, 2), info)
    
    edges = add_edge(edges, (-3, 3), info)
    edges = add_edge(edges, (-3, 5), info)
    
    edges = add_edge(edges, (-2, 3), info)
    edges = add_edge(edges, (-2, 4), info)
    
    edges = add_edge(edges, (-1, 6), info)
    
    edges = add_edge(edges, (0, 9), info)
    
    edges = add_edge(edges, (1, 7), info)
    
    edges = add_edge(edges, (2, 5), info)
    
    edges = add_edge(edges, (3, 8), info)
    
    edges = add_edge(edges, (4, 6), info)
    
    edges = add_edge(edges, (5, 7), info)
    
    edges = add_edge(edges, (6, 8), info)
    
    edges = add_edge(edges, (7, 9), info)
    edges = add_edge(edges, (7, 11), info)
    
    edges = add_edge(edges, (8, 12), info)
    
    edges = add_edge(edges, (9, 10), info)
    edges = add_edge(edges, (9, 11), info) # gated reset
    return edges, info

