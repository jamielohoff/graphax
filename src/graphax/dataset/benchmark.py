from typing import Sequence

from .utils import create, write
from ..transforms import safe_preeliminations, compress_graph, embed
from ..examples import (make_LIF, 
                        make_adaptive_LIF,
                        make_lighthouse, 
                        make_f,
                        make_g,
                        make_softmax_attention)


def make_benchmark_dataset(fname: str, info: Sequence[int] =[20, 100, 20]) -> None:
    """_summary_

    Args:
        info (GraphInfo): _description_

    Returns:
        _type_: _description_
    """
    samples = []
    create(fname, 10, info)

    # We use the field that is usually reserved for source code to store the names
    edges = make_lighthouse()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("lighthouse", edges))

    edges = make_LIF()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("LIF", edges))

    edges = make_adaptive_LIF()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("Adaptive LIF", edges))
    
    edges = make_f()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("f", edges))

    edges = make_g(size=10)
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("g", edges))
        
    edges = make_softmax_attention()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("softmax attention", edges))
    
    write(fname, samples)

    