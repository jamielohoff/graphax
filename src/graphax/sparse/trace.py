import jax
import jax.numpy as jnp

import jax.core as core


elemental_rules = {}


class CCETracer(core.Tracer):
    def __init__(self, trace, primal, elemental):
        self._trace = trace
        self.primal = primal
        self.elemental = elemental
        
    @property
    def aval(self):
        return core.get_aval(self.primal)
    

class CCETrace(core.Trace):
    pure = lift = lambda self, val: CCETracer(self, val, core.zeros_like(val)) # fix this
    
    def process_primitive(self, primitive, tracers, params):
        primal_out = primitive.bind(*[t.primal for t in tracers], **params)
        elemental_rule = elemental_rules[primitive]
        primal_out, elemental_out = elemental_rule(*[t.elemental for t in tracers], **params)
        return CCETracer(self, primal_out, elemental_out)

