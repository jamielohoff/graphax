from typing import Tuple

import chex

from ..core import GraphInfo, make_empty_edges, add_edge


def construct_Helmholtz() -> Tuple[chex.Array, GraphInfo]:
    info = GraphInfo(num_inputs=4,
                    num_intermediates=11,
                    num_outputs=4,
                    num_edges=0)
    edges = make_empty_edges(info)

    edges, info = add_edge(edges, (-3, 1), info)
    edges, info = add_edge(edges, (-3, 4), info)
    edges, info = add_edge(edges, (-3, 12), info)
    
    edges, info = add_edge(edges, (-2, 1), info)
    edges, info = add_edge(edges, (-2, 5), info)
    edges, info = add_edge(edges, (-2, 13), info)
    
    edges, info = add_edge(edges, (-1, 1), info)
    edges, info = add_edge(edges, (-1, 6), info)
    edges, info = add_edge(edges, (-1, 14), info)
    
    edges, info = add_edge(edges, (0, 1), info)
    edges, info = add_edge(edges, (0, 7), info)
    edges, info = add_edge(edges, (0, 15), info)
    
    edges, info = add_edge(edges, (1, 2), info)
    
    edges, info = add_edge(edges, (2, 3), info)
    
    edges, info = add_edge(edges, (3, 4), info)
    edges, info = add_edge(edges, (3, 5), info)
    edges, info = add_edge(edges, (3, 6), info)
    edges, info = add_edge(edges, (3, 7), info)
    
    edges, info = add_edge(edges, (4, 8), info)
    edges, info = add_edge(edges, (5, 9), info)
    edges, info = add_edge(edges, (6, 10), info)
    edges, info = add_edge(edges, (7, 11), info)
    
    edges, info = add_edge(edges, (8, 12), info)
    edges, info = add_edge(edges, (9, 13), info)
    edges, info = add_edge(edges, (10, 14), info)
    edges, info = add_edge(edges, (11, 15), info)
    return edges, info

