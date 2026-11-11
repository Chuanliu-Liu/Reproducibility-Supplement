"""Original-coordinate support certificates and exact Laplace factor absorption.

Certificates interpret stored finite binary64 observations as exact rationals.
Factor evaluation and the remaining product tree use approximate floating
arithmetic; algebraic equivalence does not certify a floating posterior sampler.
The original Compiled engine remains available as the unoptimized control.
"""
from dataclasses import replace
from fractions import Fraction
from decimal import Decimal
import math
import time
import numpy as np
from scipy.special import logsumexp
from .compiled import Compiled, NumericalConditioningError
from .model import Release
from .oracle import ImpossibleObservation


def rational(value):
    if isinstance(value, (int, np.integer, Fraction)):
        return Fraction(int(value)) if not isinstance(value, Fraction) else value
    if isinstance(value, Decimal):
        if not value.is_finite():raise ValueError('Finite rational observation required')
        return Fraction(value)
    if not isinstance(value, (float, np.floating)):
        raise ValueError('Unsupported comparison type; retain coupled likelihood')
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise ValueError('Extended precision comparison needs a separate rational certificate')
    if not math.isfinite(float(value)):
        raise ValueError('Finite observation/scale required for support certification')
    return Fraction(float(value))


def support_sign(observation, lower, upper):
    """No tolerance: uncertain/unsupported observations never authorize absorption."""
    try:
        y = rational(observation)
    except (ValueError, TypeError, OverflowError):
        return 0
    if y >= upper:
        return 1
    if y <= lower:
        return -1
    return 0


class AbsorptionCompiled:
    """Phase compiler for the original independent-Laplace, bounded q<=2 game.

    Remaining factors delegate to Compiled, including compensated numerical
    tilting and a unimodular representation of ONLY the remaining coordinates.
    q_active=0 uses class weights, with no count tensor. H remains coupled.
    """
    def __init__(self, model, release, cache=True, max_node_cells=32_000_000,
                 convolution_method='auto', query_basis=True, query_matrix=None):
        q = len(release.observed)
        if q not in (1, 2) or q != model.contributions[0].shape[1]:
            raise ValueError('Absorption compiler requires original matching q=1 or q=2')
        if release.kind not in ('laplace', 'exact'):
            raise ValueError('Unsupported likelihood: preserve the complete factor in general fallback')
        # Validate an explicitly requested ORIGINAL-coordinate representation,
        # even if absorption subsequently makes that representation unnecessary.
        if query_matrix is not None:
            from .coordinates import UnimodularCertificate
            UnimodularCertificate.create(query_matrix, q)
        self.model, self.release = model, release
        self.options = dict(cache=cache, max_node_cells=max_node_cells,
                            convolution_method=convolution_method, query_basis=query_basis)
        self.original_query_matrix = query_matrix
        self.evidence = {}
        self.generation = 0
        self.backend = None
        self._tree_totals = {}
        self.counters = dict(bound_checks=0, absorption_events=0, eliminated_coordinates=0,
                             recompilations=0, updates=0, zero_dimension_updates=0,
                             resets=0, retargets=0, sampled_worlds=0,
                             bounds_seconds=0., normalization_seconds=0.,
                             build_seconds=0., update_seconds=0., extraction_seconds=0.,
                             peak_memory_bytes=0, peak_state_cells=0)
        self.phase_seconds = {0:0., 1:0., 2:0.}
        self.reset_seconds = 0.
        self.certificates = []
        self.phase_history = []
        self._minimum = [[min(int(row[d]) for row in c) for d in range(q)]
                         for c in model.contributions]
        self._maximum = [[max(int(row[d]) for row in c) for d in range(q)]
                         for c in model.contributions]
        self._initial_lower = [sum(v[d] for v in self._minimum) for d in range(q)]
        self._initial_upper = [sum(v[d] for v in self._maximum) for d in range(q)]
        start = time.perf_counter()
        self._start_path()
        self.compile_seconds = time.perf_counter()-start

    @property
    def active_dimension(self):
        return len(self.active)

    def _archive_tree(self):
        if self.backend is not None:
            for k, v in self.backend.counters.items():
                if k.startswith('peak_'):
                    self._tree_totals[k] = max(self._tree_totals.get(k, 0), v)
                else:
                    self._tree_totals[k] = self._tree_totals.get(k, 0)+v
        self.backend = None

    def _certify(self):
        start = time.perf_counter(); newly = []
        if self.release.kind == 'laplace':
            for d in self.active:
                self.counters['bound_checks'] += 1
                sign = support_sign(self.release.observed[d], self.lower[d], self.upper[d])
                if sign:
                    scale = rational(self.release.scales[d])
                    if scale <= 0:raise ValueError('Strictly positive finite Laplace scale required')
                    self.slopes[d] = Fraction(sign)/scale
                    newly.append(d)
                    self.certificates.append(dict(generation=self.generation, coordinate=d,
                        lower=str(self.lower[d]), upper=str(self.upper[d]),
                        observed=str(rational(self.release.observed[d])),
                        slope=str(self.slopes[d]), evidence=dict(self.evidence),
                        rule='original_independent_laplace_support_sign'))
        self.active = tuple(d for d in self.active if d not in newly)
        if newly:
            self.counters['absorption_events'] += 1
            self.counters['eliminated_coordinates'] += len(newly)
        self.counters['bounds_seconds'] += time.perf_counter()-start
        return newly

    def _start_path(self):
        self.generation += 1
        self.lower, self.upper = list(self._initial_lower), list(self._initial_upper)
        self.active = tuple(range(len(self.release.observed)))
        self.slopes = {}
        self._certify()
        self._rebuild()
        self._initial_active, self._initial_slopes = self.active, dict(self.slopes)

    def _rebuild(self):
        start = time.perf_counter()
        self._archive_tree()
        self.counters['recompilations'] += 1
        self.phase_history.append(dict(generation=self.generation, prefix=len(self.evidence),
                                       active_coordinates=list(self.active)))
        # Preserve each class-dependent A_ih. Multiplying the entire product of
        # A's into pi, then using u/A for both hidden and disclosed owners, is
        # algebraically identical to the prescribed observed-u/hidden-A weights.
        logs, self.logA, self.probabilities = [], [], []
        for p, c in zip(self.model.probs, self.model.contributions):
            unary = np.zeros(len(c))
            for d, slope in self.slopes.items():
                unary += float(slope)*c[:,d]
            with np.errstate(divide='ignore'):
                logu = np.log(p)+unary[None,:]
            a = logsumexp(logu, axis=1)
            normalized = logu-a[:,None]
            prob = np.exp(normalized)
            if np.any((p > 0)&(prob == 0)) or not np.all(np.isfinite(a)):
                raise NumericalConditioningError('Absorbed unary support lost in binary64; require certified/log-domain fallback')
            logs.append(normalized); self.logA.append(a); self.probabilities.append(prob)
        self.log_probabilities = logs
        with np.errstate(divide='ignore'):
            self.base_logweights = np.log(self.model.mixture)+sum(self.logA)
        self.logweights = self.base_logweights.copy()
        for i, v in self.evidence.items():self.logweights += logs[i][:,v]
        if not np.isfinite(logsumexp(self.logweights)):
            raise ImpossibleObservation('Disclosure has zero posterior support')
        normalization_elapsed = time.perf_counter()-start
        self.counters['normalization_seconds'] += normalization_elapsed
        if self.active:
            if self.slopes:
                z = logsumexp(self.base_logweights)
                mixture = np.exp(self.base_logweights-z)
                if np.any((self.model.mixture > 0)&(mixture == 0)):
                    raise NumericalConditioningError('Absorbed class weight underflow')
                active_model = replace(self.model, probs=self.probabilities, mixture=mixture,
                    contributions=[c[:,self.active] for c in self.model.contributions])
                active_release = Release(tuple(self.release.observed[d] for d in self.active),
                    tuple(self.release.scales[d] for d in self.active), kind=self.release.kind,
                    release_id=self.release.release_id+'-absorbed-'+','.join(map(str,self.active)))
                matrix = None  # Never project a mixed old coordinate system.
            else:
                active_model, active_release = self.model, self.release
                matrix = self.original_query_matrix
            self.backend = Compiled(active_model, active_release, query_matrix=matrix, **self.options)
            if self.evidence:
                self.backend.evidence = dict(self.evidence)
                self.backend._full()
        self._record_memory()
        self.counters['build_seconds'] += time.perf_counter()-start-normalization_elapsed

    def _record_memory(self):
        self.counters['peak_memory_bytes'] = max(self.counters['peak_memory_bytes'], self.memory_bytes)
        volume = (self.backend.root.poly.size if self.backend else
                  len(self.model.mixture)*(max(self.model.target_labels)+1))
        self.counters['peak_state_cells'] = max(self.counters['peak_state_cells'], volume)

    def update(self, owner, value):
        start = time.perf_counter(); dimension = self.active_dimension
        if not 0 <= owner < len(self.model.probs) or not 0 <= value < self.model.domains[owner]:
            raise ValueError('Owner/bundle state out of domain')
        if owner == self.model.target:raise ValueError('Target cannot be a disclosure player')
        if self.evidence.get(owner) == value:return
        if owner in self.evidence:
            raise ValueError('Changing disclosure value is not prefix growth; reset first')
        proposed = self.logweights+self.log_probabilities[owner][:,value]
        if not np.isfinite(logsumexp(proposed)):
            raise ImpossibleObservation('Disclosure has zero prior/posterior support')
        self.evidence[owner] = value
        self.logweights = proposed
        bound_start = time.perf_counter()
        for d in range(len(self.release.observed)):
            v = int(self.model.contributions[owner][value,d])
            self.lower[d] += v-self._minimum[owner][d]
            self.upper[d] += v-self._maximum[owner][d]
        self.counters['bounds_seconds'] += time.perf_counter()-bound_start
        newly = self._certify()
        if newly:self._rebuild()
        elif self.backend is not None:self.backend.update(owner, value)
        else:self.counters['zero_dimension_updates'] += 1
        self.counters['updates'] += 1
        elapsed = time.perf_counter()-start
        self.counters['update_seconds'] += elapsed
        self.phase_seconds[dimension] += elapsed
        # Initial buffers dominate later dirty updates; avoid rescanning all
        # owners during a zero-dimensional O(H) update.
        if self.backend is not None:self._record_memory()

    def reset(self):
        if not self.evidence:return
        start = time.perf_counter()
        self.evidence.clear(); self.counters['resets'] += 1
        if self.active == self._initial_active and self.slopes == self._initial_slopes:
            self.generation += 1
            self.lower, self.upper = list(self._initial_lower), list(self._initial_upper)
            self.logweights = self.base_logweights.copy()
            if self.backend is not None:self.backend.reset()
        else:self._start_path()
        # A reset starts a new path. Its compilation work does not belong to
        # the previous path's final (often zero-dimensional) phase.
        self.reset_seconds += time.perf_counter()-start

    def change_release(self, release):
        if len(release.observed) != len(self.release.observed):raise ValueError('Release dimension changed')
        if release.kind not in ('laplace','exact'):raise ValueError('Unsupported complete release likelihood')
        # Validate all declared original shared-synopsis relationships once.
        release.__post_init__()
        self.release = release
        old = dict(self.evidence); self.evidence.clear(); self._start_path()
        for owner, value in old.items():self.update(owner, value)

    def retarget(self, target, labels=None):
        if target in self.evidence:raise ValueError('Cannot retarget to a disclosed owner')
        self.model = replace(self.model, target=target,
            target_labels=tuple(range(self.model.domains[target])) if labels is None else tuple(labels))
        self.generation += 1; self.counters['retargets'] += 1
        # Reestablish original certificates for this target/evidence identity.
        self.slopes = {}; self.active = tuple(range(len(self.release.observed)))
        self._certify(); self._rebuild()

    def posterior_target(self):
        start = time.perf_counter(); dimension = self.active_dimension
        if self.backend is not None:p = self.backend.posterior_target()
        else:
            z = logsumexp(self.logweights)
            if not np.isfinite(z):raise ImpossibleObservation('Impossible disclosure prefix')
            state = np.exp(self.logweights-z)@self.probabilities[self.model.target]
            p = np.bincount(self.model.target_labels, weights=state,
                            minlength=max(self.model.target_labels)+1)
            p /= p.sum()
        elapsed = time.perf_counter()-start
        self.counters['extraction_seconds'] += elapsed
        self.phase_seconds[dimension] += elapsed
        return p

    def risk(self, loss='zero_one'):
        p = self.posterior_target()
        if loss == 'zero_one':return float(1-p.max())
        if loss == 'brier':return float((1-np.square(p).sum())/2)
        raise ValueError(loss)

    def sample_world(self, rng):
        if self.evidence:raise ValueError('Complete-world draws require fresh disclosure state')
        if self.backend is not None:world = self.backend.sample_world(rng)
        else:
            weights = np.exp(self.logweights-logsumexp(self.logweights))
            h = rng.choice(len(weights), p=weights/weights.sum())
            world = np.array([rng.choice(len(p[h]), p=p[h]/p[h].sum()) for p in self.probabilities])
        self.counters['sampled_worlds'] += 1
        return world

    @property
    def tree_counters(self):
        out = dict(self._tree_totals)
        if self.backend:
            for k,v in self.backend.counters.items():
                out[k] = max(out.get(k,0),v) if k.startswith('peak_') else out.get(k,0)+v
        return out

    @property
    def memory_bytes(self):
        factors = sum(p.nbytes+l.nbytes+a.nbytes for p,l,a in
                      zip(self.probabilities,self.log_probabilities,self.logA))
        return factors+self.logweights.nbytes+self.base_logweights.nbytes+(
            self.backend.memory_bytes if self.backend else 0)

    def pruning_certificate(self, owner, include_table=False):
        # Optional oracle always sees the ORIGINAL model and release.
        from .separation import PrefixSeparation
        from .exact_tree import CertificationBudgetExceeded
        try:return PrefixSeparation(self.model,self.release).certificate(owner,self.evidence,include_table)
        except CertificationBudgetExceeded as error:
            return dict(safe_zero_current_state=False, owner=int(owner), refusal=str(error))
