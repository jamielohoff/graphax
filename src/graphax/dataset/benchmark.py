from typing import Sequence

import jax
import jax.random as jrand

from chex import PRNGKey

from .utils import create, write
from ..transforms import safe_preeliminations, compress_graph, embed
from ..examples import (make_f,
                        make_1d_roe_flux,
                        make_lif_SNN,
                        make_ada_lif_SNN,
                        make_transformer_encoder,
                        make_transformer_encoder_decoder,
                        make_lighthouse,
                        make_6DOF_robot)
                        # make_transformer_encoder_decoder)


def make_task_dataset(key: PRNGKey, fname: str, info: Sequence[int] =[20, 100, 20]) -> None:
    """
    Creates a benchmark dataset that

    Args:
        info (GraphInfo): _description_

    Returns:
        _type_: _description_
    """
    keys = jrand.split(key, 8)
    samples = []
    create(fname, 8, info)

    # We use the field that is usually reserved for source code to store the names
    edges = make_lighthouse()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[0], edges, info)
    samples.append(("Lighthouse", edges))
    
    edges = make_f()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[1], edges, info)
    samples.append(("Lighthouse", edges))

    edges = make_lif_SNN()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[2], edges, info)
    samples.append(("LIF SNN", edges))

    edges = make_ada_lif_SNN()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[3], edges, info)
    samples.append(("Adaptive LIF SNN", edges))
            
    edges = make_transformer_encoder()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[4], edges, info)
    samples.append(("Transformer Encoder", edges))
    
    edges = make_transformer_encoder_decoder()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[5], edges, info)
    samples.append(("Transformer Encoder-Decoder", edges))
    
    edges = make_1d_roe_flux()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[6], edges, info)
    samples.append(("1D Roe Flux", edges))
    
    edges = make_6DOF_robot()
    edges = safe_preeliminations(edges)
    edges = compress_graph(edges)
    edges = embed(keys[7], edges, info)
    samples.append(("Differential Kinematics 6DOF Robot", edges))
    
    write(fname, samples)

    