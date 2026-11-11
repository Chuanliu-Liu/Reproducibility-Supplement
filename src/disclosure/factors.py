"""Small general factor-world fallback independent of the scalar oracle.

Arbitrary dependent finite priors and observed-release factors supported. No
tractability promise; world budget checked BEFORE materialization. This also
keeps high-order parity likelihoods whole rather than inventing pairwise edges.
"""
import itertools
import math
import numpy as np
from scipy.special import logsumexp
from .oracle import ImpossibleObservation


class FactorWorlds:
    def __init__(self, domains, factors, target=0, max_worlds=1_048_576):
        count=math.prod(domains)
        if count>max_worlds:raise MemoryError(f'General fallback world budget exceeded: {count}')
        self.domains,self.factors,self.target=domains,factors,target
        self.worlds=np.array(list(itertools.product(*[range(d) for d in domains])),dtype=int)
        logs=np.zeros(count)
        for scope,table in factors:
            a=np.asarray(table)
            if a.shape!=tuple(domains[i] for i in scope):raise ValueError('Invalid factor scope/domain')
            with np.errstate(divide='ignore'):logs+=np.log(a[tuple(self.worlds[:,i] for i in scope)])
        z=logsumexp(logs)
        if z==-np.inf:raise ImpossibleObservation('Impossible joint factor evidence')
        self.weights=np.exp(logs-z);self.evidence={}

    def update(self,i,v):self.evidence[i]=v
    def reset(self):self.evidence.clear()
    def posterior_target(self):
        mask=np.ones(len(self.worlds),bool)
        for i,v in self.evidence.items():mask&=self.worlds[:,i]==v
        p=self.weights*mask
        if p.sum()==0:raise ImpossibleObservation('Impossible bundle evidence')
        return np.bincount(self.worlds[:,self.target],weights=p,minlength=self.domains[self.target])/p.sum()
    def risk(self,loss='zero_one'):
        p=self.posterior_target()
        return float(1-p.max()) if loss=='zero_one' else float((1-(p*p).sum())/2)
    def sample_world(self,rng):return self.worlds[rng.choice(len(self.weights),p=self.weights)].copy()
    def separated(self,owner):
        graph={i:set() for i in range(len(self.domains)) if i not in self.evidence}
        for scope,_ in self.factors:
            s=set(scope)-set(self.evidence)
            for i in s:graph[i]|=s-{i}
        seen=set();queue=[owner]
        while queue:
            i=queue.pop()
            if i not in seen:seen.add(i);queue.extend(graph.get(i,set())-seen)
        return self.target not in seen


def compile_or_fallback(model,release,max_node_cells=32_000_000,max_worlds=1_048_576,
                        likelihood_absorption=True):
    from .compiled import Compiled,StateBudgetExceeded,NumericalConditioningError
    from .baselines import Enumerated
    try:
        from .absorption import AbsorptionCompiled
        compiler = AbsorptionCompiled if likelihood_absorption else Compiled
        e=compiler(model,release,max_node_cells=max_node_cells)
        e.risk()
        return e,{'method':'absorption_compiled' if likelihood_absorption else 'compiled'}
    except (StateBudgetExceeded,NumericalConditioningError) as err:
        engine=Enumerated(model,release,max_worlds=max_worlds)
        return engine,{'method':'general_exact_enumeration_fallback','reason':str(err)}
