from .base import elemental_rules, elemental_only_rules, multi_output_elemental_only_rules

# Import submodules to trigger elemental rule registrations
# transforms must come before structural because structural imports _slice_elementals from it
from . import math
from . import linalg
from . import reductions
from . import transforms
from . import structural

from .transforms import JacobianTransform
