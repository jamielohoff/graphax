from .utils import create, write
from ..core import GraphInfo, make_graph_info
from ..transforms import safe_preeliminations_gpu, compress_graph, embed
from ..examples import (make_LIF, 
                              make_adaptive_LIF, 
                              make_Helmholtz, 
                              make_lighthouse, 
                              make_hole,
                              make_scalar_assignment_tree,
                              make_f,
                              make_g,
                              make_minimal_reverse,
                              make_hessian,
                              make_sdf_box,
                              make_sdf_sphere)


def make_benchmark_dataset(fname: str, info: GraphInfo = make_graph_info([10, 30, 5])) -> None:
    """_summary_

    Args:
        info (GraphInfo): _description_

    Returns:
        _type_: _description_
    """
    samples = []
    create(fname, 10, info)

    # We use the field that is usually reserved for source code to store the names
    edges, _info = make_lighthouse()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("lighthouse", edges, info, vertices, attn_mask))

    edges, _info = make_LIF()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("LIF", edges, info, vertices, attn_mask))

    edges, _info = make_adaptive_LIF()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("Adaptive LIF", edges, info, vertices, attn_mask))
        
    edges, _info = make_hole()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("Hole", edges, info, vertices, attn_mask))

    edges, _info = make_scalar_assignment_tree()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("Scalar assignment tree", edges, info, vertices, attn_mask))

    edges, _info = make_f()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("f", edges, info, vertices, attn_mask))

    edges, _info = make_g(size=10)
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("g", edges, info, vertices, attn_mask))

    edges, _info = make_minimal_reverse()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("minimal reverse", edges, info, vertices, attn_mask))
    
    edges, _info = make_hessian()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("hessian", edges, info, vertices, attn_mask))
    
    edges, _info = make_sdf_sphere()
    edges, _info = safe_preeliminations_gpu(edges, _info)
    edges, _info = compress_graph(edges, _info)
    edges, _, vertices, attn_mask = embed(edges, _info, info)
    samples.append(("sdf sphere", edges, info, vertices, attn_mask))
    
    write(fname, samples)

    