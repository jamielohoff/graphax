from typing import Tuple

import chex

from ..core import GraphInfo, make_graph_info, make_empty_edges, add_edge


def construct_simple() -> Tuple[chex.Array, GraphInfo]:
    info = make_graph_info([2, 2, 2])
    edges = make_empty_edges(info)
    
    edges = add_edge(edges, (-1, 1), info)
    edges = add_edge(edges, (0, 1), info)

    edges = add_edge(edges, (1, 2), info)
    edges = add_edge(edges, (1, 3), info)

    edges = add_edge(edges, (2, 3), info)
    edges = add_edge(edges, (2, 4), info)
    return edges, info

