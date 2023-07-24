from graphax.examples import make_Helmholtz
from graphax.transforms.preelimination import safe_preeliminations


edges = make_Helmholtz()

edges = safe_preeliminations(edges)
print(edges)

