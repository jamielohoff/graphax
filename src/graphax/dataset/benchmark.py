from typing import Sequence

from .utils import create, write
from ..transforms import safe_preeliminations, compress_graph, embed
from ..examples import (make_1d_roe_flux,
                        make_lif_SNN,
                        make_ada_lif_SNN,
                        make_transformer_decoder,
                        make_lighthouse,
                        make_transformer_encoder_decoder)


def make_benchmark_dataset(fname: str, info: Sequence[int] =[20, 100, 20]) -> None:
    """
    Creates a benchmark dataset that

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

    edges = make_lif_SNN()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("LIF SNN", edges))

    edges = make_ada_lif_SNN()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("Adaptive LIF SNN", edges))
            
    edges = make_transformer_decoder()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("transformer", edges))
    
    edges = make_transformer_encoder_decoder()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(edges, info)
    samples.append(("transformer", edges))
    
    write(fname, samples)

    