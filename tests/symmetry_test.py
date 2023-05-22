import jax
import jax.random as jrand

from graphax.transforms.symmetry import swap_rows, swap_cols, swap_intermediates
from graphax.examples import make_Helmholtz

edges, info = make_Helmholtz()
# print(swap_rows(2, 3, edges))
# print(swap_cols(2, 3, edges))
print(swap_intermediates(4, 5, edges, info))

