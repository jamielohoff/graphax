from typing import Tuple

import chex

from ..core import GraphInfo, make_graph_info, make_empty_edges, add_edge


def make_simple() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([2, 2, 2])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-1, 1), info)
    edges = add_edge(edges, (0, 1), info)

    edges = add_edge(edges, (1, 2), info)
    edges = add_edge(edges, (1, 3), info)

    edges = add_edge(edges, (2, 3), info)
    edges = add_edge(edges, (2, 4), info)
    return edges, info


# 15 ops
def make_lighthouse() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([4, 5, 2])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-3, 4), info)
    
    edges = add_edge(edges, (-2, 3), info)
    edges = add_edge(edges, (-2, 7), info)
    
    edges = add_edge(edges, (-1, 1), info)
    
    edges = add_edge(edges, (0, 1), info)
    
    edges = add_edge(edges, (1, 2), info)
    
    edges = add_edge(edges, (2, 3), info)
    edges = add_edge(edges, (2, 4), info)
    
    edges = add_edge(edges, (3, 5), info)
    
    edges = add_edge(edges, (4, 5), info)
    
    edges = add_edge(edges, (5, 6), info)
    edges = add_edge(edges, (5, 7), info)
    return edges, info


def make_scalar_assignment_tree() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([3, 10, 1])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-2, 1), info)
    edges = add_edge(edges, (-2, 2), info)
    
    edges = add_edge(edges, (-1, 3), info)
    edges = add_edge(edges, (-1, 4), info)
    
    edges = add_edge(edges, (0, 5), info)
    edges = add_edge(edges, (0, 6), info)
    
    edges = add_edge(edges, (1, 9), info)
    
    edges = add_edge(edges, (2, 7), info)
    
    edges = add_edge(edges, (3, 7), info)
    
    edges = add_edge(edges, (4, 8), info)
    
    edges = add_edge(edges, (5, 8), info)
    
    edges = add_edge(edges, (6, 10), info)
    
    edges = add_edge(edges, (7, 10), info)
    
    edges = add_edge(edges, (8, 9), info)
    
    edges = add_edge(edges, (9, 11), info)
    
    edges = add_edge(edges, (10, 11), info)
    return edges, info


# 34 ops
def make_hole() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([4, 5, 3])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-3, 2), info)
    
    edges = add_edge(edges, (-2, 1), info)
    
    edges = add_edge(edges, (-1, 1), info)
    
    edges = add_edge(edges, (0, 3), info)
    
    edges = add_edge(edges, (1, 2), info)
    edges = add_edge(edges, (1, 3), info)
    
    edges = add_edge(edges, (2, 4), info)
    
    edges = add_edge(edges, (3, 5), info)
    
    edges = add_edge(edges, (4, 6), info)
    edges = add_edge(edges, (4, 7), info)
    edges = add_edge(edges, (4, 8), info)
    
    edges = add_edge(edges, (5, 6), info)
    edges = add_edge(edges, (5, 7), info)
    edges = add_edge(edges, (5, 8), info)
    return edges, info

