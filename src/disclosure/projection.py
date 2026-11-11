"""C6: reuse tilted messages at support-certified absorption transitions.

Exact algebraic invariant: exp(logmass[h])*poly[h,a,z] is the
unnormalized sum of original owner factors times exp(theta[h] @ sums).
Stored z is a bijective encoding of the CURRENT active original sums.
Absorption changes theta in eliminated coordinates to the certified lambda
and marginalizes those coordinates. Float arithmetic remains approximate.
"""
from dataclasses import replace
import math
import time
import numpy as np
from scipy.special import logsumexp
from .absorption import rational, support_sign
from .compiled import Compiled, NumericalConditioningError, StateBudgetExceeded
from .oracle import ImpossibleObservation
from .input_guards import validate_factor_inputs, guard_compiled_extent


class ProjectionCompiled(Compiled):
    reset_policy = 'immutable initial snapshot; array replacement copy-on-write'
    transition_policy = 'project all off-dirty nodes; recompute dirty path; no runtime selection'

    def __init__(self, model, release, **options):
        start = time.perf_counter()
        validated = validate_factor_inputs(model, release)
        guard_compiled_extent(model, release, options, validated)
        if release.kind not in ('laplace', 'exact'):
            raise ValueError('C6 requires the unchanged independent Laplace/exact likelihood')
        q = len(release.observed)
        if q not in (1, 2):
            raise ValueError('Supported q=1 or q=2 only')
        self.options = dict(options)
        self.active = tuple(range(q))
        self.slopes = {}
        self._original_q = q
        self._ready = False
        self.metrics = dict(bound_checks=0, absorption_events=0, eliminated_coordinates=0,
            projection_entries=0, projected_nodes=0, dirty_recomputed_nodes=0,
            bounds_seconds=0., projection_seconds=0., convolution_seconds=0.,
            leaf_seconds=0., normalization_seconds=0., extraction_seconds=0.,
            reset_seconds=0., retarget_seconds=0., update_seconds=0.,
            zero_dimension_updates=0, resets=0, peak_memory_bytes=0)
        self.certificates = []
        self.phase_history = []
        self.phase_seconds = {0:0., 1:0., 2:0.}
        self._minimum = np.array([c.min(axis=0) for c in model.contributions])
        self._maximum = np.array([c.max(axis=0) for c in model.contributions])
        self._initial_lower = self._minimum.sum(axis=0)
        self._initial_upper = self._maximum.sum(axis=0)
        self.lower, self.upper = self._initial_lower.copy(), self._initial_upper.copy()
        # Preserve the original tilt/basis, initial sampler and unchanged control.
        # Initial eligible coordinates also pay construction plus conversion.
        super().__init__(model, release, **options)
        self.model_key, self.release_key = model.identity, release.identity
        self.theta = self.tilt @ self.query_basis
        self._ready = True
        newly = self._certify()
        if newly:
            self._transition(newly, None)
        self._snapshot = self._capture()
        self._snapshot_ids = {id(a) for arrays in self._snapshot['nodes'] for a in arrays}
        self._snapshot_bytes = sum(a.nbytes for arrays in self._snapshot['nodes'] for a in arrays)
        self._current_bytes = self._shared_snapshot_bytes = self._snapshot_bytes
        self._record_memory()
        self.compile_seconds = time.perf_counter()-start

    @property
    def active_dimension(self):
        return len(self.active)

    def _certify(self):
        start = time.perf_counter()
        newly = []
        if self.release.kind == 'laplace':
            for d in self.active:
                self.metrics['bound_checks'] += 1
                sign = support_sign(self.release.observed[d], int(self.lower[d]), int(self.upper[d]))
                if sign:
                    scale = rational(self.release.scales[d])
                    if scale <= 0:
                        raise ValueError('Finite positive Laplace scale required')
                    self.slopes[d] = rational(sign)/scale
                    newly.append(d)
                    self.certificates.append(dict(coordinate=d, lower=int(self.lower[d]),
                        upper=int(self.upper[d]), observed=str(rational(self.release.observed[d])),
                        slope=str(self.slopes[d]), evidence=dict(self.evidence)))
        if newly:
            self.metrics['absorption_events'] += 1
            self.metrics['eliminated_coordinates'] += len(newly)
        self.metrics['bounds_seconds'] += time.perf_counter()-start
        return newly

    def _project_node(self, node, old_active, inverse, delta, retained):
        """Decode offsets, correct class tilt, aggregate projected keys.

        Never called on a stale dirty node. Class-specific rescaling preserves
        unnormalized masses; classes with zero mass remain zero.
        """
        start = time.perf_counter()
        old = node.poly
        shape = old.shape[2:]
        stored = np.moveaxis(np.indices(shape, dtype=np.int64), 0, -1)+node.offset
        original = stored @ inverse.T
        positions = [old_active.index(d) for d in retained]
        projected = original[..., positions]
        flatkeys = projected.reshape((-1, len(retained))) if retained else None
        if retained:
            lo, hi = flatkeys.min(axis=0), flatkeys.max(axis=0)
            dims = tuple(int(x) for x in hi-lo+1)
            keys = np.ravel_multi_index((flatkeys-lo).T, dims)
        else:
            lo = np.zeros(0, dtype=np.int64)
            dims = ()
            keys = np.zeros(math.prod(shape), dtype=np.int64)
        result_shape = (old.shape[0], old.shape[1], *dims)
        if math.prod(result_shape) > self.max_node_cells:
            raise StateBudgetExceeded('Projected dense bounding box exceeds node budget')
        out = np.zeros(result_shape)
        logmass = np.full(old.shape[0], -np.inf)
        original_flat = original.reshape((-1, len(old_active)))
        for h in range(old.shape[0]):
            if not np.isfinite(node.logmass[h]):
                continue
            correction = original_flat @ delta[h]
            positive = old[h].reshape((old.shape[1], -1)) > 0
            support = positive.any(axis=0)
            if not support.any():
                continue
            shift = correction[support].max()
            weights = np.zeros_like(correction)
            weights[support] = np.exp(correction[support]-shift)
            values = old[h].reshape((old.shape[1], -1))*weights
            if np.any(positive & (values == 0)):
                raise NumericalConditioningError('C6 tilt correction loses positive represented support')
            for a in range(old.shape[1]):
                out[h,a] = np.bincount(keys, weights=values[a], minlength=math.prod(dims)).reshape(dims)
            normal_start = time.perf_counter()
            mass = out[h].sum()
            if not np.isfinite(mass) or mass <= 0:
                raise NumericalConditioningError('C6 invalid projected class mass')
            out[h] /= mass
            logmass[h] = node.logmass[h]+shift+math.log(mass)
            self.metrics['normalization_seconds'] += time.perf_counter()-normal_start
        previous = (node.poly,node.offset,node.logmass)
        node.poly, node.logmass, node.offset = out, logmass, lo
        self._account_buffers(previous,(node.poly,node.offset,node.logmass))
        self.metrics['projection_entries'] += old.size
        self.metrics['projected_nodes'] += 1
        self.metrics['projection_seconds'] += time.perf_counter()-start

    def _transition(self, newly, owner):
        old_active = self.active
        inverse = self.query_inverse.copy()
        retained = tuple(d for d in old_active if d not in newly)
        delta = np.zeros((len(self.model.mixture), len(old_active)))
        for d in newly:
            k = old_active.index(d)
            delta[:,k] = float(self.slopes[d])-self.theta[:,d]
        dirty = set()
        if owner is not None:
            n = self.leaves[owner]
            while n is not None:
                dirty.add(n.scope)
                n = n.parent
        # Off-path messages still represent the pre-event evidence correctly.
        # Project them while the old encoding and tilt are still available.
        for node in self.nodes:
            if node.scope not in dirty:
                self._project_node(node, old_active, inverse, delta, retained)
        for d in newly:
            self.theta[:,d] = float(self.slopes[d])
        self.active = retained
        self.query_basis = np.eye(len(retained), dtype=int)
        self.query_inverse = self.query_basis.copy()
        self.contributions = [c[:,retained] for c in self.model.contributions]
        self.tilt = self.theta[:,retained]
        self.likelihood_cache.clear()
        for node in self.nodes:
            if node.scope in dirty:
                self._evaluate(node)
                self.metrics['dirty_recomputed_nodes'] += 1
        self._pending_scopes.clear()
        self.phase_history.append(dict(prefix=len(self.evidence), eliminated=list(newly),
            active=list(retained), dirty_nodes=len(dirty)))

    def _evaluate(self, node):
        if not self._ready:
            return super()._evaluate(node)
        start = time.perf_counter()
        previous = (node.poly,node.offset,node.logmass)
        if node.left is None:
            i = node.scope[0]
            c, p = self.contributions[i], self.model.probs[i]
            vals = [self.evidence[i]] if i in self.evidence else np.flatnonzero(self.allowed[i])
            if i in self.evidence and not self.allowed[i][self.evidence[i]]:
                raise ImpossibleObservation('Disclosure inconsistent with exact release')
            lo, hi = c[list(vals)].min(axis=0), c[list(vals)].max(axis=0)
            t = max(self.model.target_labels)+1 if node.target else 1
            shape = (len(self.model.mixture), t, *tuple(hi-lo+1))
            if math.prod(shape) > self.max_node_cells:
                raise StateBudgetExceeded('C6 leaf state budget exceeded')
            a = np.zeros(shape)
            with np.errstate(divide='ignore'):
                logs = np.log(p)+(self.model.contributions[i] @ self.theta.T).T
            if i in self.evidence:
                v = self.evidence[i]
                node.logmass = logs[:,v].copy()
                label = self.model.target_labels[v] if node.target else 0
                a[(slice(None),label,*tuple(c[v]-lo))] = 1.
            else:
                logs[:,~self.allowed[i]] = -np.inf
                node.logmass = logsumexp(logs, axis=1)
                with np.errstate(invalid='ignore'):
                    values = np.exp(logs-node.logmass[:,None])
                values = np.nan_to_num(values, nan=0.)
                for v in vals:
                    label = self.model.target_labels[v] if node.target else 0
                    a[(slice(None),label,*tuple(c[v]-lo))] += values[:,v]
            node.poly, node.offset = a, lo
            self.counters['evaluations'] += 1
            self.metrics['leaf_seconds'] += time.perf_counter()-start
        elif self.active:
            super()._evaluate(node)
            self.metrics['convolution_seconds'] += time.perf_counter()-start
        else:
            l,r = node.left,node.right
            node.poly = l.poly*r.poly
            node.logmass = l.logmass+r.logmass
            node.offset = np.zeros(0, dtype=int)
            self.counters['evaluations'] += 1
            self.metrics['convolution_seconds'] += time.perf_counter()-start
        self._account_buffers(previous,(node.poly,node.offset,node.logmass))

    def update(self, owner, value, defer=False):
        if defer:
            raise ValueError('C6 requires eager current messages; deferred rank-certificate updates use the separate control')
        start = time.perf_counter()
        dimension = len(self.active)
        if owner == self.model.target or not 0 <= owner < len(self.model.probs):
            raise ValueError('Valid non-target owner required')
        if not 0 <= value < self.model.probs[owner].shape[1]:
            raise ValueError('Whole-bundle value outside complete domain')
        if self.evidence.get(owner) == value:
            return
        if owner in self.evidence:
            raise ValueError('Changing evidence is not a growing prefix; reset first')
        if not np.any((self.model.mixture > 0) & (self.model.probs[owner][:,value] > 0)):
            raise ImpossibleObservation('Zero supported prior mass')
        if not self.allowed[owner][value]:
            raise ImpossibleObservation('Value incompatible with exact release')
        self.evidence[owner] = value
        bstart = time.perf_counter()
        self.lower += self.model.contributions[owner][value]-self._minimum[owner]
        self.upper += self.model.contributions[owner][value]-self._maximum[owner]
        self.metrics['bounds_seconds'] += time.perf_counter()-bstart
        newly = self._certify()
        if newly:
            self._transition(newly, owner)
            self.counters['updates'] += 1
        elif self.active:
            # Existing dirty-path maintenance: no model hashes/full normalizers.
            del self.evidence[owner]
            super().update(owner, value, defer=defer)
        else:
            leaf = self.leaves[owner]
            old = leaf.logmass.copy()
            self._evaluate(leaf)
            mass = self.root.logmass.copy()
            valid = np.isfinite(mass) & np.isfinite(old) & np.isfinite(leaf.logmass)
            mass[valid] += leaf.logmass[valid]-old[valid]
            mass[~valid] = -np.inf
            self._account_buffers((self.root.logmass,),(mass,))
            self.root.logmass = mass
            self.metrics['zero_dimension_updates'] += 1
            self.counters['updates'] += 1
        elapsed = time.perf_counter()-start
        self.metrics['update_seconds'] += elapsed
        self.phase_seconds[dimension] += elapsed
        self._record_memory()

    def _weighted_root(self):
        if not self._ready:
            return super()._weighted_root()
        self._flush_pending()
        r = self.root
        self.counters['root_reads'] += 1
        self.counters['root_cells_read'] += r.poly.size
        shape = r.poly.shape[2:]
        key = (self.release_key, self.active, tuple(r.offset), shape)
        if key not in self.likelihood_cache:
            if self.active:
                grid = np.moveaxis(np.indices(shape, dtype=float),0,-1)+r.offset
                original = grid @ self.query_inverse.T
                if self.release.kind == 'laplace':
                    logs = np.sum(-np.abs(original-np.asarray(self.release.observed)[list(self.active)])/
                        np.asarray(self.release.scales)[list(self.active)]-
                        np.log(2*np.asarray(self.release.scales)[list(self.active)]),axis=-1)[None,...]
                else:
                    logs = np.where(np.all(original == np.asarray(self.release.observed)[list(self.active)],axis=-1),0.,-np.inf)[None,...]
                logs = logs-np.einsum('...q,hq->h...',grid,self.tilt)
                high = np.max(logs,axis=tuple(range(1,logs.ndim)))
                likelihood = np.exp(logs-high.reshape((-1,)+(1,)*len(shape)))
            else:
                high = np.zeros(len(self.model.mixture))
                likelihood = np.ones(len(self.model.mixture))
            # The absorbed likelihood constant is included exactly once, common
            # across classes/worlds. It matters for evidence-mass diagnostics.
            constant = sum((-float(s)*float(self.release.observed[d])-
                math.log(2*float(self.release.scales[d]))) for d,s in self.slopes.items())
            high += constant
            if len(self.likelihood_cache) >= 16:
                self.likelihood_cache.clear()
            self.likelihood_cache[key] = (likelihood, high)
        likelihood, high = self.likelihood_cache[key]
        with np.errstate(divide='ignore'):
            weights = np.log(self.model.mixture)+r.logmass+high
        shift = float(np.max(weights))
        if not np.isfinite(shift):
            raise ImpossibleObservation('Zero class/prefix mass')
        cls = np.exp(weights-shift)
        a = r.poly*likelihood[:,None]*cls.reshape((-1,1)+(1,)*len(shape))
        norm = float(a.sum())
        if not np.isfinite(norm) or norm <= 0:
            raise ImpossibleObservation('Zero release mass or numerical underflow')
        if self.counters['fft_work_proxy'] > 0 and norm < 1e-10:
            raise NumericalConditioningError('C6 FFT conditioning mass too small')
        return a/norm, math.log(norm)+shift

    def posterior_target(self):
        start = time.perf_counter()
        result = super().posterior_target()
        elapsed = time.perf_counter()-start
        self.metrics['extraction_seconds'] += elapsed
        self.phase_seconds[len(self.active)] += elapsed
        return result

    def _capture(self):
        for n in self.nodes:
            for a in (n.poly,n.offset,n.logmass):
                a.flags.writeable = False
        return dict(nodes=[(n.poly,n.offset,n.logmass) for n in self.nodes],
            active=self.active, slopes=dict(self.slopes), theta=self.theta.copy(),
            basis=self.query_basis.copy(), inverse=self.query_inverse.copy(),
            contributions=self.contributions, tilt=self.tilt.copy())

    def reset(self):
        if not self.evidence:
            return
        start = time.perf_counter()
        self.evidence.clear()
        snap = self._snapshot
        for n,buffers in zip(self.nodes,snap['nodes']):
            n.poly,n.offset,n.logmass = buffers
        self.active,self.slopes = snap['active'],dict(snap['slopes'])
        self.theta,self.query_basis,self.query_inverse = snap['theta'].copy(),snap['basis'].copy(),snap['inverse'].copy()
        self.contributions,self.tilt = snap['contributions'],snap['tilt'].copy()
        self.lower,self.upper = self._initial_lower.copy(),self._initial_upper.copy()
        self._pending_scopes.clear()
        self.likelihood_cache.clear()
        self._current_bytes = self._shared_snapshot_bytes = self._snapshot_bytes
        self.metrics['resets'] += 1
        self.metrics['reset_seconds'] += time.perf_counter()-start

    def retarget(self, target, labels=None):
        if target in self.evidence:
            raise ValueError('Retarget to undisclosed owner only')
        start = time.perf_counter()
        old = dict(self.evidence)
        model = replace(self.model,target=target,target_labels=tuple(range(self.model.domains[target])) if labels is None else tuple(labels))
        fresh = type(self)(model,self.release,**self.options)
        for i,v in old.items():
            fresh.update(i,v)
        peak = self.memory_bytes+fresh.memory_bytes
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        self.metrics['retarget_seconds'] += time.perf_counter()-start
        self.metrics['peak_memory_bytes'] = max(self.metrics['peak_memory_bytes'],peak)

    def change_release(self, release):
        old = dict(self.evidence)
        fresh = type(self)(self.model,release,**self.options)
        for i,v in old.items():
            fresh.update(i,v)
        peak = self.memory_bytes+fresh.memory_bytes
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)
        self.metrics['peak_memory_bytes'] = max(self.metrics['peak_memory_bytes'],peak)

    def sample_world(self, rng):
        # Declared original sampler fallback: no changed draw algorithm/RNG.
        if self.evidence:
            raise ValueError('Reset before drawing a complete posterior world')
        if not hasattr(self,'_original_sampler'):
            self._original_sampler = Compiled(self.model,self.release,**self.options)
        value = self._original_sampler.sample_world(rng)
        self._record_memory()
        return value

    @property
    def memory_bytes(self):
        if hasattr(self,'_snapshot_ids'):
            total = self._current_bytes+self._snapshot_bytes-self._shared_snapshot_bytes
        else:
            total = sum(n.poly.nbytes+n.offset.nbytes+n.logmass.nbytes for n in self.nodes)
        total += sum(a.nbytes+b.nbytes for a,b in self.likelihood_cache.values())
        sampler = getattr(self,'_original_sampler',None)
        return total+(sampler.memory_bytes if sampler else 0)

    def _account_buffers(self, before, after):
        if not hasattr(self,'_snapshot_ids'):
            return
        self._current_bytes += sum(a.nbytes for a in after)-sum(a.nbytes for a in before)
        self._shared_snapshot_bytes += sum(a.nbytes for a in after if id(a) in self._snapshot_ids)-sum(
            a.nbytes for a in before if id(a) in self._snapshot_ids)

    def _record_memory(self):
        # Constant node accounting plus the bounded root cache, charged to wall
        # time. Count snapshot and dirty copies without an O(n) disclosure scan.
        self.metrics['peak_memory_bytes'] = max(self.metrics['peak_memory_bytes'],self.memory_bytes)
