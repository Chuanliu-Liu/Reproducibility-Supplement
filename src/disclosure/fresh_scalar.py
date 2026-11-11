"""Lazy, cached scalar compilation at the first complete support certificate.

This control uses the same model, original-coordinate certificate, Projection
engine, initial whole-world sampler, target templates, and numerical fallback
as SuffixCompiled. Its only algorithmic difference is when scalar candidates
are prepared and maintained: before a path reaches complete absorption it
maintains no candidate class weights. At that event it prepares the one legal
pattern if necessary and folds the current prefix from its initial weights.
Subsequent disclosures cost O(H), exactly as in the eager scalar suffix.

Prepared unary tables/templates are retained across reset; they are rebuilt
only for a new target/release or a previously unseen pattern. There is no model
reparsing, hashing, sum-tree rebuild, or final projection at a successful scalar
switch. Default fast paths remain disabled, matching the existing interface.
"""
import time
import numpy as np
from .compiled import NumericalConditioningError
from .oracle import ImpossibleObservation
from .suffix import SuffixCompiled, _index


class FreshScalarCompiled(SuffixCompiled):
    """Strong on-demand comparator for the eager all-pattern scalar bank."""

    def __init__(self, model, release, enable_fast_paths=False, **options):
        # The parent owns input admission, initial aliases, inner construction,
        # reset/rebuild, fallback, and state transactionality. Its mode-builder
        # hook skips only noninitial candidates during this constructor.
        self._defer_initial_candidates = True
        try:
            super().__init__(model, release,
                             enable_fast_paths=enable_fast_paths, **options)
        finally:
            self._defer_initial_candidates = False
        self.counters.update(fresh_preparations=0, fresh_folds=0,
                             fresh_fold_rows=0, fresh_prepare_seconds=0.,
                             fresh_fold_seconds=0.)

    def _make_mode(self, mode):
        if (getattr(self, '_defer_initial_candidates', False)
                and self._initial_mode is None):
            return
        return super()._make_mode(mode)

    def _prepare_fresh_weights(self, mode, owner, value):
        """Build/cache one original-factor pattern and fold this growing prefix."""
        if mode not in self._bank:
            started = time.perf_counter()
            self._make_mode(mode)
            self._retained_bank_bytes = sum(
                a.nbytes for pattern in self._bank for a in
                (self._bank[pattern], self._initial_bank[pattern],
                 self._templates[pattern], *self._logprob[pattern]))
            self.counters['fresh_preparations'] += 1
            self.counters['fresh_prepare_seconds'] += time.perf_counter() - started
        if mode in self._bad_modes:
            raise NumericalConditioningError('Fresh scalar pattern unavailable')
        started = time.perf_counter()
        weights = self._initial_bank[mode].copy()
        tables = self._logprob[mode]
        for previous_owner, previous_value in self.evidence.items():
            weights += tables[previous_owner][:, previous_value]
        weights += tables[owner][:, value]
        self.counters['fresh_folds'] += 1
        self.counters['fresh_fold_rows'] += len(self.evidence) + 1
        self.counters['fresh_fold_seconds'] += time.perf_counter() - started
        return weights

    def update(self, owner, value, defer=False):
        # Once scalar, the parent performs the identical O(H) weight update.
        # In particular, it owns numerical fallback and poison/reset behavior.
        self._check_healthy()
        if self._mode is not None:
            return super().update(owner, value, defer=defer)
        start = time.perf_counter()
        if defer:
            raise ValueError('FreshScalar requires eager updates')
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
        contribution = tuple(int(v) for v in self.model.contributions[owner][value])
        lo = tuple(a + v - m for a, v, m in
                   zip(self.lower, contribution, self._mins[owner]))
        hi = tuple(a + v - m for a, v, m in
                   zip(self.upper, contribution, self._maxs[owner]))
        signs = self._signs(lo, hi)
        switch = False
        weights = None
        try:
            if (self.enable_fast_paths and all(signs)
                    and signs not in self._bad_modes):
                try:
                    weights = self._prepare_fresh_weights(signs, owner, value)
                    self._weights_to_probs(weights, live)
                    switch = True
                except NumericalConditioningError:
                    # Same supported fallback: retain the ordinary Projection
                    # representation if this scalar calculation loses mass.
                    pass
            if not switch:
                self.inner.update(owner, value)
                if self.release.kind == 'exact':
                    self.inner.posterior_target()
        except Exception as exc:
            # Unexpected allocation/inner errors may have mutated hidden state.
            # Refuse all further use until the inherited reset has rebuilt it.
            self._poison(exc)
            raise
        self.lower, self.upper, self._live = lo, hi, live
        self.evidence[owner] = value
        if switch:
            self._bank[signs] = weights
            self._mode = signs
            self.counters['switches'] += 1
        self.counters['updates'] += 1
        self.counters['update_seconds'] += time.perf_counter() - start
