import sys
from .interpreter.from_jaxpr import make_graph
from .core import (make_empty_edges, 
                    vertex_eliminate, 
                    forward, 
                    reverse)
from .vertex_game import VertexGameState, VertexGame, make_vertex_game_state
from .transforms.embedding import embed

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

