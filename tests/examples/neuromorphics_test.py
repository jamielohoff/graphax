from graphax.examples import make_SNN
from graphax.transforms.markowitz import minimal_markowitz
from graphax import forward, reverse, cross_country

edges = make_SNN()
print(edges)

order = minimal_markowitz(edges)
print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

