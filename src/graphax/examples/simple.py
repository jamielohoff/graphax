from typing import Tuple

import chex

from ..core import GraphInfo, make_empty_edges, add_edge


def construct_simple() -> Tuple[chex.Array, GraphInfo]:
    info = GraphInfo(num_inputs=2,
                    num_intermediates=2,
                    num_outputs=2,
                    num_edges=0)
    edges = make_empty_edges(info)
    
    edges, info = add_edge(edges, (-1, 1), info)
    edges, info = add_edge(edges, (0, 1), info)

    edges, info = add_edge(edges, (1, 2), info)
    edges, info = add_edge(edges, (1, 3), info)

    edges, info = add_edge(edges, (2, 3), info)
    edges, info = add_edge(edges, (2, 4), info)
    return edges, info

