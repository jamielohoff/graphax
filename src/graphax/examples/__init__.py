from .helmholtz import make_Helmholtz, make_free_energy
from .simple import (make_simple, 
                    make_lighthouse, 
                    make_scalar_assignment_tree, 
                    make_hole)
from .neuromorphic import make_LIF, make_adaptive_LIF, make_SNN
from .advanced import make_f, make_g, make_minimal_reverse, make_hessian
from .random_codegenerator import make_random_code
from .differential_rendering import make_sdf_box, make_sdf_sphere
from .deep_learning import make_softmax_attention, make_Perceptron, make_transformer_encoder