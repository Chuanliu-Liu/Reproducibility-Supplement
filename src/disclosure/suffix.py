"""Support-certified scalar suffix for original independent Laplace releases.

Candidate tables use original factors and enter inference only after complete
support certification. Inputs are fixed for an instance; use retarget and
change_release for supported changes. Float arithmetic remains approximate.
Unexpected inner-engine failures poison the instance until a successful reset.
"""
from dataclasses import replace
from itertools import product
import operator
import sys
import time
import numpy as np
from scipy.special import logsumexp
from .absorption import support_sign
from .compiled import NumericalConditioningError, StateBudgetExceeded
from .input_guards import validate_factor_inputs, guard_compiled_extent
from .oracle import ImpossibleObservation
from .projection import ProjectionCompiled


class SuffixStateError(RuntimeError):
    """An interrupted inner update requires reset before further use."""


class _AliasTable:
    """Immutable, preprocessed finite categorical sampler (float arithmetic)."""
    def __init__(self, probability):
        p = np.asarray(probability, dtype=float)
        if (p.ndim != 1 or not len(p) or not np.all(np.isfinite(p))
                or np.any(p < 0) or not np.isfinite(p.sum()) or p.sum() <= 0):
            raise NumericalConditioningError('Invalid categorical probabilities')
        p = p / p.sum()
        n = len(p)
        scaled = p * n
        self.threshold = np.ones(n, dtype=float)
        self.alias = np.arange(n, dtype=np.intp)
        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            a, b = small.pop(), large.pop()
            self.threshold[a] = scaled[a]
            self.alias[a] = b
            remainder = scaled[b] - (1.0 - scaled[a])
            if remainder < -64 * np.finfo(float).eps:
                raise NumericalConditioningError('Unstable categorical preprocessing')
            scaled[b] = max(0.0, remainder)
            (small if scaled[b] < 1.0 else large).append(b)
        self.threshold.flags.writeable = False
        self.alias.flags.writeable = False

    def draw(self, rng):
        slot = int(rng.integers(len(self.threshold)))
        return slot if float(rng.random()) < self.threshold[slot] else int(self.alias[slot])

    @property
    def nbytes(self):
        return self.threshold.nbytes + self.alias.nbytes


def _bound_bytes(value):
    if isinstance(value, tuple):
        return sys.getsizeof(value) + sum(_bound_bytes(x) for x in value)
    return sys.getsizeof(value)


def _index(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be an integer index')
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(name + ' must be an integer index') from exc


class SuffixCompiled:
    def __init__(self, model, release, enable_fast_paths=False, **options):
        start = time.perf_counter()
        data = validate_factor_inputs(model, release)
        # Initial scalar mode must not silently round unsupported large integers.
        # This scans factors and candidate coordinates, never building a sum tree.
        guard_compiled_extent(model, release, options, data)
        self.model, self.release = model, release
        self.enable_fast_paths, self.options = bool(enable_fast_paths), dict(options)
        self._inputs = data
        self.q, self._domains = data.q, data.domains
        self._mins, self._maxs = data.mins, data.maxs
        self._lo0, self._hi0 = data.lower, data.upper
        self.lower, self.upper = self._lo0, self._hi0
        self._support_bytes = sum(_bound_bytes(x) for x in
                                  (self._mins, self._maxs, self._lo0, self._hi0))
        self.evidence = {}
        self._prior_live = np.asarray(model.mixture > 0, dtype=bool)
        self._live = self._prior_live.copy()
        self._poisoned = False
        self._poison_reason = ''
        self.counters = dict(updates=0, resets=0, switches=0, sampled_worlds=0,
                             fallback=0, update_seconds=0., reset_seconds=0.,
                             rebuild_seconds=0., retargets=0, release_changes=0)
        if data.h * (max(data.labels) + 1) > options.get('max_node_cells', 32_000_000):
            raise StateBudgetExceeded('Target template exceeds declared cell budget')
        self.inner = None
        self._bank, self._initial_bank = {}, {}
        self._templates, self._logprob = {}, {}
        self._bad_modes = set()
        self._class_alias = None
        self._owner_aliases = None
        self._alias_bytes = 0
        signs = self._signs(self.lower, self.upper)
        self._initial_mode = signs if (self.enable_fast_paths and all(signs)
                                      and release.kind == 'laplace') else None
        self._mode = self._initial_mode
        if release.kind == 'laplace':
            modes = [self._initial_mode] if self._initial_mode else list(product((-1, 1), repeat=self.q))
            for mode in modes:
                self._make_mode(mode)
            if self._initial_mode:
                try:
                    if self._initial_mode in self._bad_modes:
                        raise NumericalConditioningError('Initial scalar pattern unavailable')
                    self._prepare_aliases(self._initial_mode)
                except NumericalConditioningError:
                    self._initial_mode = self._mode = None
                    self._class_alias = self._owner_aliases = None
                    self._alias_bytes = 0
                    for mode in product((-1, 1), repeat=self.q):
                        if mode not in self._bank:
                            self._make_mode(mode)
                    self.counters['fallback'] += 1
        if self._mode is None:
            self.inner = self._build_inner({})
        self._retained_bank_bytes = sum(a.nbytes for mode in self._bank for a in
            (self._bank[mode], self._initial_bank[mode], self._templates[mode], *self._logprob[mode]))
        self.compile_seconds = time.perf_counter() - start

    def _check_healthy(self):
        if self._poisoned:
            raise SuffixStateError('Reset required after failed inner operation: ' + self._poison_reason)

    def _poison(self, error):
        self._poisoned = True
        self._poison_reason = type(error).__name__

    def _signs(self, lower, upper):
        if self.release.kind != 'laplace':
            return (0,) * self.q
        return tuple(support_sign(y, lo, hi) for y, lo, hi in
                     zip(self.release.observed, lower, upper))

    @staticmethod
    def _weights_to_probs(weights, live=None):
        if live is not None and np.any(live & ~np.isfinite(weights)):
            raise NumericalConditioningError('Lost a supported scalar class weight')
        z = logsumexp(weights)
        if not np.isfinite(z):
            raise NumericalConditioningError('Scalar class weights are not normalizable')
        p = np.exp(weights - z)
        if (not np.all(np.isfinite(p)) or p.sum() <= 0
                or np.any(np.isfinite(weights) & (p == 0))):
            raise NumericalConditioningError('Scalar class normalization lost positive mass')
        return p / p.sum()

    def _make_mode(self, mode):
        h = self._inputs.h
        normalized = [np.full((h, d), -np.inf) for d in self._domains]
        weights = np.full(h, -np.inf)
        template = np.zeros((h, max(self._inputs.labels) + 1))
        bad = False
        with np.errstate(divide='ignore', invalid='ignore', over='ignore', under='ignore'):
            slope = np.asarray(mode, dtype=float) / np.asarray(self.release.scales, dtype=float)
            if not np.all(np.isfinite(slope)):
                bad = True
            else:
                weights[self._prior_live] = np.log(self.model.mixture[self._prior_live])
                for i, (p, c) in enumerate(zip(self.model.probs, self.model.contributions)):
                    linear = np.asarray(c, dtype=float) @ slope
                    if not np.all(np.isfinite(linear)):
                        bad = True
                        break
                    u = np.log(p[self._prior_live]) + linear[None, :]
                    a = logsumexp(u, axis=1)
                    if not np.all(np.isfinite(a)):
                        bad = True
                        break
                    v = u - a[:, None]
                    if (np.any(np.isnan(v)) or np.any(np.isposinf(v))
                            or np.any(np.isfinite(v) & (np.exp(v) == 0))):
                        bad = True
                        break
                    normalized[i][self._prior_live] = v
                    weights[self._prior_live] += a
                if not np.all(np.isfinite(weights[self._prior_live])):
                    bad = True
            if not bad:
                target = normalized[self.model.target]
                for state, label in enumerate(self._inputs.labels):
                    template[:, label] += np.exp(target[:, state])
        if bad:
            self._bad_modes.add(mode)
            weights.fill(-np.inf)
            for v in normalized:
                v.fill(-np.inf)
            template.fill(0.0)
        self._logprob[mode] = normalized
        self._bank[mode] = weights
        self._initial_bank[mode] = weights.copy()
        self._templates[mode] = template

    def _prepare_aliases(self, mode):
        class_probability = self._weights_to_probs(self._bank[mode], self._prior_live)
        class_alias = _AliasTable(class_probability)
        owners = []
        for h in range(self._inputs.h):
            if not self._prior_live[h]:
                owners.append(None)
                continue
            owners.append(tuple(_AliasTable(np.exp(v[h])) for v in self._logprob[mode]))
        self._class_alias = class_alias
        self._owner_aliases = tuple(owners)
        self._alias_bytes = class_alias.nbytes + sum(
            a.nbytes for row in owners if row is not None for a in row)

    def _build_inner(self, evidence):
        # Keep this admission check even when the C6 entry point also checks it.
        # No unsafe input reaches the ordinary encoder or its allocations.
        guard_compiled_extent(self.model, self.release, self.options, self._inputs)
        fresh = ProjectionCompiled(self.model, self.release, **self.options)
        for owner, value in evidence.items():
            fresh.update(owner, value)
        fresh.posterior_target()  # Require positive, numerically evaluable evidence.
        return fresh

    @property
    def active(self):
        self._check_healthy()
        return () if self._mode is not None else self.inner.active

    @property
    def active_dimension(self):
        return len(self.active)

    def _commit_fallback(self, fresh):
        self.inner = fresh
        self._mode = self._initial_mode = None
        self.enable_fast_paths = False
        self._class_alias = self._owner_aliases = None
        self._alias_bytes = 0
        self.counters['fallback'] += 1

    def update(self, owner, value, defer=False):
        self._check_healthy()
        start = time.perf_counter()
        if defer:
            raise ValueError('Suffix requires eager updates')
        owner, value = _index(owner, 'Owner'), _index(value, 'Bundle state')
        if not 0 <= owner < self._inputs.n or owner == self.model.target:
            raise ValueError('Valid non-target owner required')
        if not 0 <= value < self._domains[owner]:
            raise ValueError('Whole-bundle value outside complete domain')
        if self.evidence.get(owner) == value:
            return
        if owner in self.evidence:
            raise ValueError('Changing evidence is not a growing prefix; reset first')
        live = self._live & (self.model.probs[owner][:, value] > 0)
        if not np.any(live):
            raise ImpossibleObservation('Disclosure has zero joint owner/class support')
        modes = [self._mode] if self._mode is not None else list(self._bank)
        proposed = {s: self._bank[s] + self._logprob[s][owner][:, value] for s in modes}
        contribution = tuple(int(v) for v in self.model.contributions[owner][value])
        lo = tuple(a + v - m for a, v, m in zip(self.lower, contribution, self._mins[owner]))
        hi = tuple(a + v - m for a, v, m in zip(self.upper, contribution, self._maxs[owner]))
        signs = self._signs(lo, hi)
        switch = False
        if (self._mode is None and self.enable_fast_paths and all(signs)
                and signs not in self._bad_modes):
            try:
                self._weights_to_probs(proposed[signs], live)
                switch = True
            except NumericalConditioningError:
                pass
        fallback = False
        if self._mode is not None:
            try:
                self._weights_to_probs(proposed[self._mode], live)
            except NumericalConditioningError:
                fallback = True
        try:
            if fallback:
                # Build through the proposed prefix before replacing live state.
                extended = dict(self.evidence)
                extended[owner] = value
                fresh = self._build_inner(extended)
                self._commit_fallback(fresh)
            elif self._mode is None and not switch:
                self.inner.update(owner, value)
                if self.release.kind == 'exact':
                    self.inner.posterior_target()
        except Exception as exc:
            self._poison(exc)
            raise
        for mode, weights in proposed.items():
            self._bank[mode] = weights
        self.lower, self.upper, self._live = lo, hi, live
        self.evidence[owner] = value
        if switch:
            self._mode = signs
            self.counters['switches'] += 1
        self.counters['updates'] += 1
        self.counters['update_seconds'] += time.perf_counter() - start

    def posterior_target(self):
        self._check_healthy()
        try:
            if self._mode is None:
                return self.inner.posterior_target()
            try:
                weights = self._weights_to_probs(self._bank[self._mode], self._live)
                result = weights @ self._templates[self._mode]
                if not np.all(np.isfinite(result)) or result.sum() <= 0:
                    raise NumericalConditioningError('Invalid scalar target posterior')
                return result / result.sum()
            except NumericalConditioningError:
                fresh = self._build_inner(self.evidence)
                self._commit_fallback(fresh)
                return self.inner.posterior_target()
        except Exception as exc:
            self._poison(exc)
            raise

    def risk(self, loss='zero_one'):
        if loss not in ('zero_one', 'brier'):
            raise ValueError(loss)
        p = self.posterior_target()
        return float(1 - p.max()) if loss == 'zero_one' else float((1 - np.square(p).sum()) / 2)

    def sample_world(self, rng):
        self._check_healthy()
        if self.evidence:
            raise ValueError('Reset before drawing a complete posterior world')
        try:
            if self._initial_mode is None:
                result = self.inner.sample_world(rng)
            else:
                h = self._class_alias.draw(rng)
                result = np.fromiter((table.draw(rng) for table in self._owner_aliases[h]),
                                     dtype=int, count=self._inputs.n)
        except Exception as exc:
            self._poison(exc)
            raise
        self.counters['sampled_worlds'] += 1
        return result

    def reset(self):
        start = time.perf_counter()
        if self._poisoned:
            # A failed first update may leave outer evidence empty: always rebuild.
            fresh = type(self)(self.model, self.release,
                               enable_fast_paths=self.enable_fast_paths, **self.options)
            counters = dict(self.counters)
            counters['resets'] += 1
            counters['reset_seconds'] += time.perf_counter() - start
            fresh.counters = counters
            fresh.compile_seconds += self.compile_seconds
            self.__dict__.clear()
            self.__dict__.update(fresh.__dict__)
            return
        if not self.evidence:
            return
        try:
            if self.inner is not None:
                self.inner.reset()
        except Exception as exc:
            self._poison(exc)
            raise
        for mode in self._bank:
            self._bank[mode] = self._initial_bank[mode].copy()
        self.lower, self.upper = self._lo0, self._hi0
        self._live = self._prior_live.copy()
        self.evidence.clear()
        self._mode = self._initial_mode
        self.counters['resets'] += 1
        self.counters['reset_seconds'] += time.perf_counter() - start

    def _rebuild(self, model, release, event):
        self._check_healthy()
        start = time.perf_counter()
        old = dict(self.evidence)
        fresh = type(self)(model, release, enable_fast_paths=self.enable_fast_paths, **self.options)
        for owner, value in old.items():
            fresh.update(owner, value)
        fresh.posterior_target()
        counters = {k: self.counters[k] + fresh.counters[k] for k in self.counters}
        counters[event] += 1
        counters['rebuild_seconds'] += time.perf_counter() - start
        fresh.counters = counters
        fresh.compile_seconds += self.compile_seconds
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    def change_release(self, release):
        self._rebuild(self.model, release, 'release_changes')

    def retarget(self, target, labels=None):
        self._check_healthy()
        target = _index(target, 'Target owner')
        if not 0 <= target < self._inputs.n or target in self.evidence:
            raise ValueError('Retarget to an undisclosed valid owner only')
        labels = tuple(range(self._domains[target])) if labels is None else tuple(labels)
        model = replace(self.model, target=target, target_labels=labels)
        self._rebuild(model, self.release, 'retargets')

    @property
    def memory_bytes(self):
        """Retained numeric payload and Python integer bounds; measure RSS separately."""
        return ((self.inner.memory_bytes if self.inner is not None else 0)
                + self._retained_bank_bytes + self._alias_bytes + self._support_bytes
                + _bound_bytes(self.lower) + _bound_bytes(self.upper)
                + self._prior_live.nbytes + self._live.nbytes)
