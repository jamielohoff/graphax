from typing import Tuple

import chex

from ..core import GraphInfo, make_graph_info, make_empty_edges, add_edge

# 33 ops
def make_Helmholtz() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([4, 11, 4])
    edges = make_empty_edges(info)

    edges = add_edge(edges, (-3, 1), info)
    edges = add_edge(edges, (-3, 4), info)
    edges = add_edge(edges, (-3, 12), info)
    
    edges = add_edge(edges, (-2, 1), info)
    edges = add_edge(edges, (-2, 5), info)
    edges = add_edge(edges, (-2, 13), info)
    
    edges = add_edge(edges, (-1, 1), info)
    edges = add_edge(edges, (-1, 6), info)
    edges = add_edge(edges, (-1, 14), info)
    
    edges = add_edge(edges, (0, 1), info)
    edges = add_edge(edges, (0, 7), info)
    edges = add_edge(edges, (0, 15), info)
    
    edges = add_edge(edges, (1, 2), info)
    
    edges = add_edge(edges, (2, 3), info)
    
    edges = add_edge(edges, (3, 4), info)
    edges = add_edge(edges, (3, 5), info)
    edges = add_edge(edges, (3, 6), info)
    edges = add_edge(edges, (3, 7), info)
    
    edges = add_edge(edges, (4, 8), info)
    edges = add_edge(edges, (5, 9), info)
    edges = add_edge(edges, (6, 10), info)
    edges = add_edge(edges, (7, 11), info)
    
    edges = add_edge(edges, (8, 12), info)
    edges = add_edge(edges, (9, 13), info)
    edges = add_edge(edges, (10, 14), info)
    edges = add_edge(edges, (11, 15), info)
    return edges, info


def make_free_energy():
    info = make_graph_info([4, 15, 1])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-3, 1), info)
    edges = add_edge(edges, (-3, 4), info)
    edges = add_edge(edges, (-3, 12), info)
    
    edges = add_edge(edges, (-2, 1), info)
    edges = add_edge(edges, (-2, 5), info)
    edges = add_edge(edges, (-2, 13), info)
    
    edges = add_edge(edges, (-1, 1), info)
    edges = add_edge(edges, (-1, 6), info)
    edges = add_edge(edges, (-1, 14), info)
    
    edges = add_edge(edges, (0, 1), info)
    edges = add_edge(edges, (0, 7), info)
    edges = add_edge(edges, (0, 15), info)
    
    edges = add_edge(edges, (1, 2), info)
    
    edges = add_edge(edges, (2, 3), info)
    
    edges = add_edge(edges, (3, 4), info)
    edges = add_edge(edges, (3, 5), info)
    edges = add_edge(edges, (3, 6), info)
    edges = add_edge(edges, (3, 7), info)
    
    edges = add_edge(edges, (4, 8), info)
    edges = add_edge(edges, (5, 9), info)
    edges = add_edge(edges, (6, 10), info)
    edges = add_edge(edges, (7, 11), info)
    
    edges = add_edge(edges, (8, 12), info)
    edges = add_edge(edges, (9, 13), info)
    edges = add_edge(edges, (10, 14), info)
    edges = add_edge(edges, (11, 15), info)
    
    edges = add_edge(edges, (12, 16), info)
    edges = add_edge(edges, (13, 16), info)
    edges = add_edge(edges, (14, 16), info)
    edges = add_edge(edges, (15, 16), info)
    return edges, info
