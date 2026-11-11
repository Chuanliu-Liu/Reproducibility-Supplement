"""Offline evaluation of the ORIGINAL disclosure prefixes in descending order.

No sampler, absorption, likelihood-coordinate factorization, or online interface.
Complete bundle probabilities enter evidence weights separately from aggregate
factors. Coefficients use nonnegative shift/add; the full joint release is applied
once per target-state contraction. Float diagnostics are not certificates.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
import time
import numpy as np
from .compiled import NumericalConditioningError, StateBudgetExceeded
from .coordinates import UnimodularCertificate
from .oracle import ImpossibleObservation
from .estimate import PathBudgetExceeded


@dataclass
class ReversePath:
    increments: np.ndarray
    risks: np.ndarray
    # Actual bundle-state posterior, before aggregation by validated target label.
    target_posteriors: object = None


class ReverseSuffix:
    def __init__(self, model, release, *, players=None, matrix=None, tilt=None,
                 backend='dense', dtype='float64', tile_cells=16384,
                 max_coefficient_bytes=512*1024*1024, admission=None):
        start = time.perf_counter()
        self.model, self.release = model, release
        self.q, self.h = model.contributions[0].shape[1], len(model.mixture)
        if self.q not in (1, 2) or self.q != len(release.observed):
            raise ValueError('RSC supports the declared additive integer q1/q2 family')
        if backend not in ('dense', 'sparse') or tile_cells < 1:
            raise ValueError('Invalid kernel/tile')
        self.backend, self.dtype = backend, np.dtype(dtype)
        if self.dtype not in (np.dtype('float64'), np.dtype('longdouble')):
            raise ValueError('Use float64 or longdouble; no low precision support thinning')
        self.tile_cells, self.max_bytes, self.admission = tile_cells, max_coefficient_bytes, admission
        if any(not isinstance(v, (int, np.integer)) or v < 0 for v in model.target_labels):
            raise ValueError('Validated nonnegative integer target labels required')
        self.players = tuple(i for i in range(len(model.probs)) if i != model.target) if players is None else tuple(players)
        if len(set(self.players)) != len(self.players) or any(i == model.target or i < 0 or i >= len(model.probs) for i in self.players):
            raise ValueError('Invalid players; the target is excluded')
        self.fixed = tuple(i for i in range(len(model.probs)) if i != model.target and i not in self.players)
        identity = tuple(tuple(int(i == j) for j in range(self.q)) for i in range(self.q))
        self.coordinates = UnimodularCertificate.create(identity if matrix is None else matrix, self.q)
        self.contributions = [np.asarray([self.coordinates.encode(row) for row in c], dtype=np.int64) for c in model.contributions]
        self.inverse = np.asarray(self.coordinates.inverse, dtype=np.int64)
        self.tilt = np.zeros((self.h, self.q), dtype=self.dtype) if tilt is None else np.asarray(tilt, dtype=self.dtype).copy()
        if self.tilt.shape != (self.h, self.q) or not np.isfinite(self.tilt).all():
            raise ValueError('Fixed public/model/release tilt required')
        # p'=p exp(theta.f)/Z. ALL owner Z factors enter the class prefactor,
        # including the target and any permanently hidden nonplayers.
        self.probs, self.log_probs, all_log_z = [], [], np.zeros(self.h, dtype=self.dtype)
        for p, c in zip(model.probs, self.contributions):
            p = np.asarray(p, dtype=self.dtype)
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite prior')
            with np.errstate(divide='ignore'):
                logs = np.log(p)+(c@self.tilt.T).T
            hi = logs.max(axis=1)
            exp = self._exp(logs-hi[:, None], p > 0)
            z = exp.sum(axis=1)
            log_z = hi+np.log(z)
            tilted = exp/z[:, None]
            self._check_positive(tilted, p > 0, 'tilted owner state')
            self.probs.append(tilted)
            self.log_probs.append(logs-log_z[:, None])
            all_log_z += log_z
        with np.errstate(divide='ignore'):
            self.class_log_prefactor = np.log(np.asarray(model.mixture, dtype=self.dtype))+all_log_z
        self.factors = []
        for p, c in zip(self.probs, self.contributions):
            grouped = {}
            for v, s in enumerate(c):
                key = tuple(map(int, s))
                if key not in grouped:
                    grouped[key] = np.zeros(self.h, dtype=self.dtype)
                grouped[key] += p[:, v]
            self.factors.append(tuple((s, p) for s, p in sorted(grouped.items())))
        self.identity = (model.identity, release.identity, model.target)
        self.kernel_identity = hashlib.sha256(json.dumps(dict(matrix=self.coordinates.matrix,
            tilt=self.tilt.tolist(), backend=self.backend, dtype=str(self.dtype), tile_cells=self.tile_cells),
            sort_keys=True,default=float).encode()).hexdigest()
        self.counters = dict(paths=0, resets=0, stencil_products=0, coefficient_allocated_cells=0,
                             likelihood_cells=0, contractions=0, peak_two_buffer_bytes=0,
                             fixed_base_products=0, prefix_probability_reads=0, zero_class_revivals=0)
        self.phase_seconds = dict(prefix_metadata=0., coefficient_updates=0., likelihood_contractions=0., reset=0.)
        self.compile_seconds = time.perf_counter()-start

    def _check_positive(self, values, positive, where):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(positive & (values == 0)):
            raise NumericalConditioningError('Lost positive/nonfinite mass in '+where)

    def _exp(self, logs, positive):
        with np.errstate(under='ignore', over='ignore', invalid='ignore'):
            values = np.exp(logs)
        self._check_positive(values, positive, 'exponential')
        return values

    @property
    def memory_bytes(self):
        # Model, likelihood, sampler, path outputs and metadata charged separately.
        return sum(p.nbytes+l.nbytes+c.nbytes for p, l, c in zip(self.probs, self.log_probs, self.contributions))+self.tilt.nbytes+self.class_log_prefactor.nbytes+sum(p.nbytes for factor in self.factors for _, p in factor)

    def _admit(self, needed):
        if needed > self.max_bytes:
            raise StateBudgetExceeded(f'RSC two coefficient buffers need {needed} bytes > {self.max_bytes}')
        if self.admission is not None:
            self.admission(needed)
        self.counters['peak_two_buffer_bytes'] = max(self.counters['peak_two_buffer_bytes'], needed)

    def _base(self):
        start = time.perf_counter()
        if self.backend == 'dense':
            table, offset = np.ones((self.h,)+(1,)*self.q, dtype=self.dtype), np.zeros(self.q, dtype=np.int64)
        else:
            table, offset = {tuple([0]*self.q): np.ones(self.h, dtype=self.dtype)}, None
        for i in self.fixed:
            before = self.counters['stencil_products']
            table, offset = self._add(table, offset, i)
            self.counters['fixed_base_products'] += self.counters['stencil_products']-before
        self.counters['resets'] += 1
        self.phase_seconds['reset'] += time.perf_counter()-start
        return table, offset

    def _add(self, table, offset, owner):
        factor = self.factors[owner]
        if self.backend == 'sparse':
            # Charge actual Python storage conservatively before expansion.
            self._admit((len(table)+len(table)*len(factor))*(256+self.h*self.dtype.itemsize))
            out = {}
            for s, a in table.items():
                for f, p in factor:
                    value = a*p
                    self._check_positive(value, (a > 0) & (p > 0), 'sparse product')
                    key = tuple(x+y for x, y in zip(s, f))
                    if key not in out:
                        out[key] = np.zeros(self.h, dtype=self.dtype)
                    out[key] += value
                    self.counters['stencil_products'] += self.h
            self.counters['coefficient_allocated_cells'] += len(out)*self.h
            return out, None
        low = np.min([s for s, _ in factor], axis=0)
        high = np.max([s for s, _ in factor], axis=0)
        dims = tuple(int(a+b-c) for a, b, c in zip(table.shape[1:], high, low))
        cells = self.h*math.prod(dims)
        # Product temp is tiled, never an extra full coefficient array.
        self._admit(table.nbytes+cells*self.dtype.itemsize+self.h*self.tile_cells*self.dtype.itemsize)
        out = np.zeros((self.h,)+dims, dtype=self.dtype)
        self.counters['coefficient_allocated_cells'] += cells
        for s, p in factor:
            shift = np.asarray(s)-low
            region = out[(slice(None), *[slice(int(v), int(v)+d) for v, d in zip(shift, table.shape[1:])])]
            # Tile along the leading coordinate; one row is also tiled for q2.
            for h in range(self.h):
                if p[h] == 0:  # Algebraic exact zero only.
                    continue
                if self.q == 1:
                    for lo in range(0, table[h].size, self.tile_cells):
                        a = table[h, lo:lo+self.tile_cells]
                        value = a*p[h]
                        self._check_positive(value, a > 0, 'dense product')
                        region[h, lo:lo+self.tile_cells] += value
                else:
                    width = table.shape[2]
                    rows = max(1, self.tile_cells//width)
                    for lo in range(0, table.shape[1], rows):
                        for col in range(0, width, self.tile_cells):
                            a = table[h, lo:lo+rows, col:col+self.tile_cells]
                            value = a*p[h]
                            self._check_positive(value, a > 0, 'dense product')
                            region[h, lo:lo+rows, col:col+self.tile_cells] += value
                self.counters['stencil_products'] += table[h].size
        if not np.isfinite(out).all():
            raise NumericalConditioningError('Nonfinite coefficient sum')
        return out, offset+low

    def _tiles(self, table, offset):
        if self.backend == 'sparse':
            keys = list(table)
            for lo in range(0, len(keys), self.tile_cells):
                subset = keys[lo:lo+self.tile_cells]
                yield np.asarray(subset, dtype=np.int64), np.stack([table[k] for k in subset], axis=1)
        else:
            dims = table.shape[1:]
            flat = table.reshape(self.h, -1)
            for lo in range(0, flat.shape[1], self.tile_cells):
                index = np.arange(lo, min(lo+self.tile_cells, flat.shape[1]))
                coords = np.stack(np.unravel_index(index, dims), axis=1)+offset
                yield coords, flat[:, lo:lo+self.tile_cells]

    def _target(self, table, offset, t, prefix_logs, zeros):
        logs = np.full(self.model.domains[self.model.target], -np.inf, dtype=self.dtype)
        for a, f in enumerate(self.contributions[self.model.target]):
            for coords, coeff in self._tiles(table, offset):
                total = coords+t+f
                # The adapter receives the ORIGINAL complete joint count vector.
                original = total@self.inverse.T
                lk = np.asarray(self.release.log_likelihood(original), dtype=self.dtype)
                if np.isnan(lk).any() or np.isposinf(lk).any():
                    raise NumericalConditioningError('Invalid joint log likelihood')
                self.counters['likelihood_cells'] += len(coords)
                for h in range(self.h):
                    if zeros[h] or self.probs[self.model.target][h, a] == 0 or not np.isfinite(self.class_log_prefactor[h]):
                        continue
                    active = (coeff[h] > 0) & np.isfinite(lk)
                    if not active.any():
                        continue
                    term = np.log(coeff[h, active])+lk[active]-total[active]@self.tilt[h]
                    high = term.max()
                    integral = high+np.log(self._exp(term-high, np.ones(term.size, dtype=bool)).sum())
                    mass = self.class_log_prefactor[h]+prefix_logs[h]+self.log_probs[self.model.target][h, a]+integral
                    logs[a] = np.logaddexp(logs[a], mass)
        if not np.isfinite(logs).any():
            raise ImpossibleObservation('Release/disclosure has zero normalization')
        hi = logs.max()
        mass = self._exp(logs-hi, np.isfinite(logs))
        posterior = mass/mass.sum()
        self._check_positive(posterior, np.isfinite(logs), 'target normalization')
        self.counters['contractions'] += 1
        return posterior

    def _risk(self, posterior, loss):
        label_mass = np.zeros(max(self.model.target_labels)+1, dtype=self.dtype)
        for a, label in enumerate(self.model.target_labels):
            label_mass[label] += posterior[a]
        if loss == 'zero_one':
            return 1-label_mass.max()
        if loss == 'brier':
            return (1-np.square(label_mass).sum())/2
        raise ValueError(loss)

    def evaluate_path(self, world, order, loss='zero_one', *, posteriors=False, deadline=None, resume=None, checkpoint=None):
        order = tuple(map(int, order))
        if len(world) != len(self.model.probs) or len(order) != len(self.players) or set(order) != set(self.players):
            raise ValueError('A permutation of exactly the original players is required')
        for i in order:
            if not 0 <= int(world[i]) < self.model.domains[i]:
                raise ValueError('Disclosed bundle state outside its domain')
        # x_j and values of fixed hidden owners are deliberately never read.
        n = len(order)
        started = time.perf_counter()
        if resume is None:
            t = sum((self.contributions[i][int(world[i])] for i in order), start=np.zeros(self.q, dtype=np.int64))
            prefix_logs = np.zeros(self.h, dtype=self.dtype)
            zeros = np.zeros(self.h, dtype=np.int64)
            for i in order:
                values = self.log_probs[i][:, int(world[i])]
                positive = np.isfinite(values)
                zeros += ~positive
                prefix_logs[positive] += values[positive]
                self.counters['prefix_probability_reads'] += self.h
            risks = np.full(n+1, np.nan, dtype=self.dtype)
            states = [None]*(n+1) if posteriors else None
            k = n
        else:
            if resume['identity'] != self.identity or resume.get('kernel_identity') != self.kernel_identity or resume['order'] != order or resume['evidence_values'] != tuple(int(world[i]) for i in order) or resume['loss'] != loss:
                raise ValueError('Stale partial RSC path')
            table, offset, t, prefix_logs, zeros, risks, states, k = (resume[x] for x in ('table','offset','t','prefix_logs','zeros','risks','states','next_k'))
        self.phase_seconds['prefix_metadata'] += time.perf_counter()-started
        if resume is None:
            table, offset = self._base()
        while k >= 0:
            if deadline is not None and time.perf_counter() >= deadline:
                partial = dict(identity=self.identity, kernel_identity=self.kernel_identity, order=order, evidence_values=tuple(int(world[i]) for i in order), loss=loss,
                               table=table, offset=offset, t=t, prefix_logs=prefix_logs, zeros=zeros,
                               risks=risks, states=states, next_k=k, completed_prefixes=n-k)
                if checkpoint is not None:
                    checkpoint(partial)
                raise PathBudgetExceeded(partial)
            started = time.perf_counter()
            posterior = self._target(table, offset, t, prefix_logs, zeros)
            risks[k] = self._risk(posterior, loss)
            if states is not None:
                states[k] = posterior.copy()
            self.phase_seconds['likelihood_contractions'] += time.perf_counter()-started
            if k == 0:
                break
            i = order[k-1]
            started = time.perf_counter()
            table, offset = self._add(table, offset, i)
            self.phase_seconds['coefficient_updates'] += time.perf_counter()-started
            t = t-self.contributions[i][int(world[i])]
            values = self.log_probs[i][:, int(world[i])]
            positive = np.isfinite(values)
            old = zeros.copy()
            zeros -= ~positive
            self.counters['zero_class_revivals'] += int(((old > 0) & (zeros == 0)).sum())
            prefix_logs[positive] -= values[positive]
            if k == 1:
                prefix_logs[:] = 0  # Exact empty evidence, not residual subtraction roundoff.
            k -= 1
        increments = np.zeros(len(world), dtype=self.dtype)
        for k, i in enumerate(order, 1):
            increments[i] = risks[k-1]-risks[k]
        if not np.isclose(increments.sum(), risks[0]-risks[-1], atol=1e-12, rtol=1e-12):
            raise NumericalConditioningError('RSC telescoping failure')
        self.counters['paths'] += 1
        return ReversePath(increments, risks, states)

    def reset(self):
        # Each call starts from its own base. No suffix cache survives a path.
        self.counters['resets'] += 1

    def retarget(self, target, labels=None):
        model = replace(self.model, target=target, target_labels=tuple(range(self.model.domains[target])) if labels is None else tuple(labels))
        full_game = set(self.players) == set(range(len(self.model.probs)))-{self.model.target}
        players = None if full_game else tuple(i for i in self.players if i != target)
        return type(self)(model, self.release, players=players, matrix=self.coordinates.matrix, tilt=self.tilt,
                          backend=self.backend, dtype=self.dtype, tile_cells=self.tile_cells,
                          max_coefficient_bytes=self.max_bytes, admission=self.admission)

    def change_release(self, release, *, tilt=None):
        # New release may use a new explicitly supplied fixed tilt. No stale samples.
        return type(self)(self.model, release, players=self.players, matrix=self.coordinates.matrix, tilt=tilt,
                          backend=self.backend, dtype=self.dtype, tile_cells=self.tile_cells,
                          max_coefficient_bytes=self.max_bytes, admission=self.admission)


class SequentialFromScratch(ReverseSuffix):
    """Matched PGF/stencil control: independently rebuild EACH forward prefix."""
    def evaluate_path(self, world, order, loss='zero_one', *, posteriors=False, deadline=None, **kwargs):
        order = tuple(map(int, order))
        if len(world) != len(self.model.probs) or len(order) != len(self.players) or set(order) != set(self.players):
            raise ValueError('Invalid original player permutation')
        risks, states = [], [] if posteriors else None
        t = np.zeros(self.q, dtype=np.int64)
        logs, zeros = np.zeros(self.h, dtype=self.dtype), np.zeros(self.h, dtype=np.int64)
        for k in range(len(order)+1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise PathBudgetExceeded(dict(next_prefix=k, risks=[float(r) for r in risks], world=list(map(int, world)), order=order))
            table, offset = self._base()
            for i in reversed(order[k:]):
                table, offset = self._add(table, offset, i)
            posterior = self._target(table, offset, t, logs, zeros)
            risks.append(self._risk(posterior, loss))
            if states is not None:
                states.append(posterior)
            if k < len(order):
                i = order[k]
                t += self.contributions[i][int(world[i])]
                values = self.log_probs[i][:, int(world[i])]
                positive = np.isfinite(values)
                logs[positive] += values[positive]
                zeros += ~positive
        increments = np.zeros(len(world), dtype=self.dtype)
        for k, i in enumerate(order, 1):
            increments[i] = risks[k-1]-risks[k]
        self.counters['paths'] += 1
        return ReversePath(increments, np.asarray(risks), states)
