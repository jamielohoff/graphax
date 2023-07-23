from graphax import cross_country
from graphax.examples import make_Helmholtz
from graphax.transforms.markowitz import minimal_markowitz


edges = make_Helmholtz()

order = minimal_markowitz(edges)
print(order)

print(cross_country(order, edges))

