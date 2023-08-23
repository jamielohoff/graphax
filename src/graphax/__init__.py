import sys
from .core import (make_empty_edges, 
                    vertex_eliminate,
                    cross_country, 
                    forward, 
                    reverse,
                    get_shape,
                    get_elimination_order,
                    get_output_mask,
                    get_vertex_mask)
from .vertex_game import step
from .transforms import (embed,
                        clean,
                        compress,
                        safe_preeliminations,
                        minimal_markowitz)
from .interpreter import make_graph, jacve

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

