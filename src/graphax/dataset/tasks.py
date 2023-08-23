from typing import Sequence

import jax
import jax.random as jrand

from chex import PRNGKey

from .utils import create, write
from ..transforms import safe_preeliminations, compress, embed
from ..examples import (make_f,
                        make_1d_roe_flux,
                        make_lif_SNN,
                        make_ada_lif_SNN,
                        make_transformer_encoder,
                        make_transformer_encoder_decoder,
                        make_lighthouse,
                        make_6DOF_robot)


def make_task_dataset(key: PRNGKey, fname: str, info: Sequence[int] =[20, 105, 20]) -> None:
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
    
    # Number of FMAs after safe preeliminations
    # fwd: 16, rev: 11, cc: 13
    edges = make_lighthouse()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[0], edges, info)
    samples.append(("Lighthouse", edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 19, rev: 13, cc: 15
    edges = make_f()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[1], edges, info)
    samples.append(("f", edges))
            
    # Number of FMAs after safe preeliminations
    # fwd: 69824, rev: n/v, cc: 23552
    edges = make_transformer_encoder()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[4], edges, info)
    samples.append(("Transformer Encoder", edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 81968, rev: 15584, cc: 32400
    edges = make_transformer_encoder_decoder()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[5], edges, info)
    samples.append(("Transformer Encoder-Decoder", edges))
    
    # Number of FMAs after safe preeliminations
    # fwd: 384, rev: 217, cc: 277
    edges = make_1d_roe_flux()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[6], edges, info)
    samples.append(("1D Roe Flux", edges))
    
    #  Number of FMAs after safe preeliminations
    # fwd: 329, rev: 177, cc: 181
    edges = make_6DOF_robot()
    edges = safe_preeliminations(edges)
    edges = compress(edges)
    edges = embed(keys[7], edges, info)
    samples.append(("Differential Kinematics 6DOF Robot", edges))
    
    write(fname, samples)    

    