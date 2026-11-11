"""Same-game full recomputation and SAGE-style conditional risk access."""
import itertools
import math
import numpy as np
from scipy.special import logsumexp
from .oracle import ImpossibleObservation


class Enumerated:
    """Independent vectorized full-joint exact fallback, never advertised as scalable."""
    def __init__(self, model, release, max_worlds=1_048_576):
        count = math.prod(model.domains)
        if count > max_worlds:
            raise MemoryError(f'Exact fallback requires {count} worlds > declared budget {max_worlds}')
        self.model, self.release = model, release
        self.worlds = np.array(list(itertools.product(*[range(d) for d in model.domains])), dtype=int)
        with np.errstate(divide='ignore'):
            logs = np.log(model.mixture)[None, :]+np.zeros((count,len(model.mixture)))
            for i, p in enumerate(model.probs):
                logs += np.log(p[:,self.worlds[:,i]]).T
        c = np.zeros((count,len(release.observed)))
        for i, contrib in enumerate(model.contributions):
            c += contrib[self.worlds[:,i]]
        weights = logsumexp(logs,axis=1)+release.log_likelihood(c)
        z = logsumexp(weights)
        if z == -np.inf:
            raise ImpossibleObservation('Impossible release')
        self.weights = np.exp(weights-z)
        self.evidence = {}

    def update(self, i, value):
        self.evidence[i] = value

    def reset(self):
        self.evidence.clear()

    def posterior_target(self):
        mask = np.ones(len(self.worlds),dtype=bool)
        for i,v in self.evidence.items():
            mask &= self.worlds[:,i] == v
        w = self.weights*mask
        if w.sum() == 0:
            raise ImpossibleObservation('Impossible disclosure')
        labels = np.asarray(self.model.target_labels)[self.worlds[:,self.model.target]]
        return np.bincount(labels,weights=w,minlength=max(labels)+1)/w.sum()

    def risk(self,loss='zero_one'):
        p = self.posterior_target()
        return float(1-p.max()) if loss == 'zero_one' else float((1-np.square(p).sum())/2)

    def sample_world(self,rng):
        return self.worlds[rng.choice(len(self.weights),p=self.weights)].copy()


class FullRecompute:
    """Rebuild all sum messages at each evidence update, same compact model."""
    def __init__(self, model, release):
        from .compiled import Compiled
        self.engine = Compiled(model,release,cache=False)

    def __getattr__(self,k):
        return getattr(self.engine,k)


class SageConditional:
    """Faithful conditional Bayes-risk permutation baseline; not default SAGE imputer.

    Uses the exact same posterior oracle as full-recompute for fair isolation of
    prefix reuse. Complements sampled conditionally on disclosed full bundles;
    Bayes prediction is analytically integrated (Rao-Blackwellized conditional
    imputation). This is a reimplementation, not a claim that official SAGE was
    executed with its marginal imputer.
    """
    def __init__(self, model, release):
        from .compiled import Compiled
        self.engine = Compiled(model, release, cache=False)

    def __getattr__(self,k):
        return getattr(self.engine,k)
