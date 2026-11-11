"""Explicit store-all suffix-DP comparator, separate from the rolling engine.

This is the same PGF algebra with precomputed suffix tables; it is not claimed
to be an inferior asymptotic algorithm or a different statistical estimator.
All retained tables are charged. The rolling suffix-DP schedule is RSC itself.
"""
import time
import numpy as np
from .reverse_suffix import ReverseSuffix, ReversePath
from .compiled import StateBudgetExceeded
from .estimate import PathBudgetExceeded


class StoredSuffixDP(ReverseSuffix):
    def evaluate_path(self, world, order, loss='zero_one', *, posteriors=False, deadline=None, **kwargs):
        order = tuple(map(int, order))
        if len(order) != len(self.players) or set(order) != set(self.players) or len(world) != len(self.model.probs):
            raise ValueError('Invalid original prefix path')
        tables = [None]*(len(order)+1)
        tables[-1] = self._base()
        retained = tables[-1][0].nbytes
        for k in range(len(order)-1, -1, -1):
            if deadline is not None and time.perf_counter() >= deadline:
                raise PathBudgetExceeded(dict(stage='suffix-dp-compilation', next_k=k, original_order=order))
            current, offset = tables[k+1]
            low = np.min([s for s, _ in self.factors[order[k]]], axis=0)
            high = np.max([s for s, _ in self.factors[order[k]]], axis=0)
            next_bytes = self.h*int(np.prod(np.asarray(current.shape[1:])+high-low))*self.dtype.itemsize
            self._admit(retained+next_bytes+self.h*self.tile_cells*self.dtype.itemsize)
            tables[k] = self._add(current, offset, order[k])
            retained += tables[k][0].nbytes
        self.counters['stored_suffix_retained_peak_bytes'] = max(self.counters.get('stored_suffix_retained_peak_bytes', 0), retained)
        t, logs, zeros = np.zeros(self.q, dtype=np.int64), np.zeros(self.h, dtype=self.dtype), np.zeros(self.h, dtype=np.int64)
        risks, states = [], [] if posteriors else None
        for k, (table, offset) in enumerate(tables):
            if deadline is not None and time.perf_counter() >= deadline:
                raise PathBudgetExceeded(dict(stage='suffix-dp-forward-contractions', next_prefix=k, risks=[float(x) for x in risks]))
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
