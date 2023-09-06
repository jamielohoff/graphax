import jax
import jax.numpy as np
from jax import random
from jax_md import energy, quantity, space

box_size = 25.0
displacement_fn, shift_fn = space.periodic(box_size)

N = 1000
spatial_dimension = 2
key = random.PRNGKey(0)
R = random.uniform(key, (N, spatial_dimension), minval=0.0, maxval=1.0)
energy_fn = energy.lennard_jones_pair(displacement_fn)
print(jax.make_jaxpr(energy_fn)(R))
print('E = {}'.format(energy_fn(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))

