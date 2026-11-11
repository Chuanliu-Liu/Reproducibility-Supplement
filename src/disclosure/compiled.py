"""Finite-sum convolution DAG. Float direct/FFT arithmetic is approximate.

Supported compact family: independent owners conditional on a finite latent class,
bounded integer contribution vectors compiled from public-key relational joins.
Two target channels preserve the hidden target value during all convolutions.
Each node carries normalized class-conditional polynomials and logarithmic evidence
mass; observed owners collapse to point contributions. Exact backward count-state
sampling draws COMPLETE worlds from the observed-release posterior.
"""
from dataclasses import dataclass
from dataclasses import replace
import itertools
import math
import time
import numpy as np
from scipy.signal import convolve, fftconvolve
from scipy.special import logsumexp
from scipy.optimize import minimize
from .oracle import ImpossibleObservation
from .coordinates import UnimodularCertificate, contribution_box


class StateBudgetExceeded(RuntimeError):
    pass


class NumericalConditioningError(ArithmeticError):
    pass


@dataclass
class Node:
    scope: tuple
    left: object = None
    right: object = None
    parent: object = None
    poly: object = None
    offset: object = None
    logmass: object = None
    target: bool = False


class Compiled:
    def __init__(self, model, release, cache=True, max_node_cells=32_000_000,
                 convolution_method='auto', query_basis=True, query_matrix=None):
        start = time.perf_counter()
        self.model, self.release, self.cache = model, release, cache
        if len(release.observed) > 2 or len(release.observed) != model.contributions[0].shape[1]:
            raise ValueError('Supported compact compiler requires matching q<=2')
        if convolution_method not in ('auto', 'direct', 'fft'):
            raise ValueError('Unknown convolution method')
        self.convolution_method = convolution_method
        self.use_query_basis = query_basis
        self.public_query_matrix = query_matrix
        self.key = (model.identity, release.identity, model.target)
        self.evidence = {}
        self._pending_scopes = set()
        self._separation = None
        self._prepare_query_basis()
        self._prepare_release()
        self.max_node_cells = max_node_cells
        self.nodes, self.leaves = [], {}
        self.counters = dict(evaluations=0, convolution_direct_multiply_adds=0,
                             fft_work_proxy=0., peak_node_cells=0, updates=0,
                             affected_nodes=0, sampled_worlds=0)
        self.counters.update(root_reads=0, root_cells_read=0,
                             dense_output_cells=0, sparse_support_cells=0)
        self.counters.update(deferred_updates=0, materialized_dirty_nodes=0)
        def build(lo, hi):
            node = Node(tuple(range(lo, hi)), target=lo <= model.target < hi)
            if hi-lo > 1:
                mid = (lo+hi)//2
                node.left, node.right = build(lo, mid), build(mid, hi)
                node.left.parent = node.right.parent = node
            else:
                self.leaves[lo] = node
            self.nodes.append(node)
            return node
        self.root = build(0, len(model.probs))
        self._full()
        self.compile_seconds = time.perf_counter()-start
        self.structure = [{"scope": n.scope, "shape": n.poly.shape,
                           "cells": n.poly.size, "has_target_channel": n.target}
                          for n in self.nodes]
        self.likelihood_cache = {}

    def _prepare_query_basis(self):
        """Reversible integer coordinates for overlapping public query sums.

        Common and contrast sums can be much narrower than two highly overlapping
        raw counts. Preserve the original synopsis likelihood jointly in decoded
        coordinates; do not pretend contrast noises are independent.
        """
        q=len(self.release.observed)
        candidates=[np.eye(q,dtype=int)]
        if self.public_query_matrix is not None:
            candidates=[UnimodularCertificate.create(self.public_query_matrix, q).matrix]
        elif q==2 and self.use_query_basis:
            candidates.extend([np.array([[1,0],[-1,1]]),np.array([[1,-1],[0,1]])])
        choices=[]
        for matrix in candidates:
            certificate=UnimodularCertificate.create(matrix, q)
            contrib=[np.array([certificate.encode(row) for row in c],dtype=int)
                     for c in self.model.contributions]
            box=contribution_box(contrib)
            choices.append((box['volume'],certificate,contrib,box['widths']))
        cells,certificate,contrib,widths=min(choices,key=lambda x:x[0])
        self.query_basis=np.array(certificate.matrix,dtype=int)
        self.query_inverse=np.array(certificate.inverse,dtype=int)
        self.contributions=contrib
        self.basis_diagnostic={'matrix':self.query_basis.tolist(),'inverse':self.query_inverse.tolist(),
                               'certificate':certificate.as_dict(),
                               'original_sum_box_cells':contribution_box(self.model.contributions)['volume'],
                               'compiled_sum_box_cells':cells,
                               'compiled_widths':list(widths),
                               'likelihood':'original full synopsis likelihood evaluated on decoded sums; no independent contrast-noise factorization'}

    def _prepare_release(self):
        """Algebraic leaf tilting for Laplace observations outside sum support.

        log L(c) = t.c + residual(c); multiply leaf probabilities by exp(t.c_i)
        and subtract t.c once from root likelihood. This preserves the FULL
        observed-release posterior while avoiding FFT-tail-dominated conditioning.
        Exact boundary counts imply every owner attains its corresponding bound;
        impossible states are removed by a sufficient support constraint.
        """
        q=len(self.release.observed)
        hcount=len(self.model.mixture)
        self.tilt=np.zeros((hcount,q))
        self.allowed=[np.ones(len(c),dtype=bool) for c in self.contributions]
        low=np.sum([c.min(axis=0) for c in self.contributions],axis=0)
        high=np.sum([c.max(axis=0) for c in self.contributions],axis=0)
        y=self.query_basis@np.asarray(self.release.observed)
        slope_bounds=np.abs(self.query_inverse).T@(1/np.asarray(self.release.scales))
        if self.release.kind=='laplace':
            self.tilt[:]=np.where(y<=low,-slope_bounds,np.where(y>=high,slope_bounds,0))
        elif self.release.kind=='exact':
            if np.any(y<low) or np.any(y>high):
                raise ImpossibleObservation('Exact count outside declared owner sum support')
            for i,c in enumerate(self.contributions):
                for d in range(q):
                    if y[d]==low[d]:self.allowed[i]&=c[:,d]==c[:,d].min()
                    if y[d]==high[d]:self.allowed[i]&=c[:,d]==c[:,d].max()
                if not self.allowed[i].any():
                    raise ImpossibleObservation('Incompatible exact boundary counts')
        # Interior rare observations also require stable mass placement. Optimize
        # an exponential tilt of the complete latent-class prior. Any returned t
        # preserves the posterior algebraically; convergence affects stability,
        # not the definition of the model/game. Laplace slope bounds prevent a
        # residual likelihood concentrating on unrepresented numerical tails.
        active=[np.flatnonzero(a) for a in self.allowed]
        target=np.clip(y,low+1e-8,high-1e-8)
        bounds=[(-s,s) for s in slope_bounds] if self.release.kind=='laplace' else [(-40.,40.)]*q
        diagnostics=[]
        for h in range(hcount):
            def objective(t):
                logz=0.;mean=np.zeros(q)
                for p,c,vals in zip(self.model.probs,self.contributions,active):
                    with np.errstate(divide='ignore'):logs=np.log(p[h,vals])+c[vals]@t
                    z=logsumexp(logs)
                    if z==-np.inf:return 1e300,np.zeros(q)
                    logz+=z;mean+=np.exp(logs-z)@c[vals]
                return float(logz-t@target),mean-target
            opt=minimize(objective,self.tilt[h],method='L-BFGS-B',jac=True,bounds=bounds,
                         options={'maxiter':30,'ftol':1e-12,'gtol':1e-7})
            self.tilt[h]=opt.x
            diagnostics.append({'h':h,'iterations':int(opt.nit),'optimizer_success':bool(opt.success),'tilt':opt.x.tolist()})
        self.tilt_diagnostic={'method':'class-specific bounded log-partition tilting; exact reparameterization with class-specific likelihood compensation','classes':diagnostics}

    def _evaluate(self, node):
        self.counters['evaluations'] += 1
        if node.left is None:
            i = node.scope[0]
            c, p = self.contributions[i], self.model.probs[i]
            vals = [self.evidence[i]] if i in self.evidence else np.flatnonzero(self.allowed[i])
            if i in self.evidence and not self.allowed[i][self.evidence[i]]:
                raise ImpossibleObservation('Disclosure inconsistent with exact release support')
            lo = np.min(c[list(vals)], axis=0)
            hi = np.max(c[list(vals)], axis=0)
            t = max(self.model.target_labels)+1 if node.target else 1
            shape = (len(self.model.mixture), t, *tuple(hi-lo+1))
            cells = math.prod(shape)
            if cells > self.max_node_cells:
                raise StateBudgetExceeded(f"leaf cells {cells}")
            a = np.zeros(shape)
            if i in self.evidence:
                v = self.evidence[i]
                with np.errstate(divide='ignore'):
                    node.logmass = np.log(p[:, v])+self.tilt@c[v]
                label = self.model.target_labels[v] if node.target else 0
                a[(slice(None), label, *tuple(c[v]-lo))] = 1.
            else:
                with np.errstate(divide='ignore'):
                    logs=np.log(p)+(c@self.tilt.T).T
                logs[:,~self.allowed[i]]=-np.inf
                node.logmass=logsumexp(logs,axis=1)
                with np.errstate(invalid='ignore'):
                    tilted=np.exp(logs-node.logmass[:,None])
                tilted=np.nan_to_num(tilted,nan=0.)
                for v in vals:
                    label = self.model.target_labels[v] if node.target else 0
                    a[(slice(None), label, *tuple(c[v]-lo))] += tilted[:, v]
            node.poly, node.offset = a, lo
        else:
            l, r = node.left, node.right
            t = max(l.poly.shape[1], r.poly.shape[1])
            dims = tuple(a+b-1 for a, b in zip(l.poly.shape[2:], r.poly.shape[2:]))
            shape = (len(self.model.mixture), t, *dims)
            cells = math.prod(shape)
            if cells > self.max_node_cells:
                raise StateBudgetExceeded(f"scope {len(node.scope)}: {cells} cells > {self.max_node_cells}")
            a = np.zeros(shape)
            for h in range(len(self.model.mixture)):
                for v in range(t):
                    x = l.poly[h, v if l.target else 0]
                    y = r.poly[h, v if r.target else 0]
                    cost = x.size*y.size
                    if self.convolution_method == 'direct' or (self.convolution_method == 'auto' and cost <= 4096):
                        z = convolve(x, y, mode='full', method='direct')
                        self.counters['convolution_direct_multiply_adds'] += cost
                    else:
                        z = fftconvolve(x, y, mode='full')
                        # Floating FFT roundoff only, never clip attribution increments.
                        if z.min() < -1e-10:
                            raise ArithmeticError('Negative probability beyond FFT roundoff')
                        z = np.maximum(z, 0)
                        self.counters['fft_work_proxy'] += math.prod(dims)*math.log2(max(2, math.prod(dims)))
                    a[h, v] = z
            node.poly = a
            node.offset = l.offset+r.offset
            node.logmass = l.logmass+r.logmass
        self.counters['peak_node_cells'] = max(self.counters['peak_node_cells'], node.poly.size)
        self.counters['dense_output_cells'] += node.poly.size
        self.counters['sparse_support_cells'] += int(np.count_nonzero(node.poly))

    def _full(self):
        for n in self.nodes:
            self._evaluate(n)
        self._pending_scopes.clear()

    def _flush_pending(self):
        if self._pending_scopes:
            for node in self.nodes:  # Existing postorder; children before parents.
                if node.scope in self._pending_scopes:
                    self._evaluate(node)
                    self.counters['materialized_dirty_nodes'] += 1
            self._pending_scopes.clear()

    def update(self, owner, value, defer=False):
        if not 0 <= value < self.model.probs[owner].shape[1]:
            raise ValueError('Bundle state out of domain')
        if self.evidence.get(owner) == value:
            return
        self.evidence[owner] = value
        dirty = []
        node = self.leaves[owner]
        while node is not None:
            dirty.append(node)
            node = node.parent
        self.counters['updates'] += 1
        self.counters['affected_nodes'] += len(dirty)
        if self.cache:
            self._pending_scopes.update(n.scope for n in dirty)
            if defer:
                self.counters['deferred_updates'] += 1
            else:
                self._flush_pending()
        else:
            self._full()

    def update_certified_zero(self, owner, value, certificate):
        if (not certificate.get('safe_zero_current_state') or
                certificate.get('rule') != 'exact_release_conditioned_joint_rank_one' or
                certificate.get('model') != self.model.identity or
                certificate.get('release') != self.release.identity or
                certificate.get('evidence') != self.evidence or
                certificate.get('owner') != owner):
            raise ValueError('Missing or stale prefix-specific certificate')
        if value not in certificate['supported_values']:
            raise ImpossibleObservation('Disclosure outside certified positive support')
        self.update(owner, value, defer=True)

    def reset(self):
        if self.evidence:
            self.evidence.clear()
            self._full()

    def change_release(self, release):
        self.release = release
        self.key = (self.model.identity, release.identity, self.model.target)
        self.likelihood_cache.clear()
        self._prepare_release()
        self._full()

    def retarget(self, target, labels=None):
        """Reuse target-free subcircuits; rebuild union of old/new target paths."""
        if target in self.evidence:
            raise ValueError('Cannot retarget to an already disclosed hidden variable')
        self._flush_pending()
        old=self.model.target
        labels=tuple(range(self.model.domains[target])) if labels is None else tuple(labels)
        self.model=replace(self.model,target=target,target_labels=labels)
        dirty=set()
        for owner in [old,target]:
            node=self.leaves[owner]
            while node is not None:
                dirty.add(node.scope);node=node.parent
        for node in self.nodes:
            node.target=target in node.scope
            if node.scope in dirty:self._evaluate(node)
        self.key=(self.model.identity,self.release.identity,target)
        self.counters['retargets']=self.counters.get('retargets',0)+1
        self.counters['retarget_affected_nodes']=self.counters.get('retarget_affected_nodes',0)+len(dirty)

    def _weighted_root(self):
        self._flush_pending()
        r = self.root
        self.counters['root_reads'] += 1
        self.counters['root_cells_read'] += r.poly.size
        shape = r.poly.shape[2:]
        lk = (self.release.identity, tuple(r.offset), shape)
        if lk not in self.likelihood_cache:
            grid = np.moveaxis(np.indices(shape, dtype=float), 0, -1)+r.offset
            original_grid=grid@self.query_inverse.T
            logs = self.release.log_likelihood(original_grid)[None,...]-np.einsum('...q,hq->h...',grid,self.tilt)
            high = np.max(logs,axis=tuple(range(1,logs.ndim)))
            if np.all(high == -np.inf):
                raise ImpossibleObservation('Observed release impossible under remaining support')
            likelihood = np.exp(logs-high.reshape((-1,)+(1,)*len(shape)))
            # Bounded cache memory across prefixes; identities include exact offsets/domains.
            if len(self.likelihood_cache) >= 16:
                self.likelihood_cache.clear()
            self.likelihood_cache[lk] = (likelihood, high)
        likelihood, high = self.likelihood_cache[lk]
        with np.errstate(divide='ignore'):
            weights = np.log(self.model.mixture)+r.logmass+high
        shift = float(np.max(weights))
        if shift == -np.inf:
            raise ImpossibleObservation('Disclosure has zero prior/posterior support')
        cls = np.exp(weights-shift)
        a = r.poly*likelihood[:,None]*cls.reshape((-1, 1)+ (1,)*len(shape))
        norm = float(a.sum())
        if norm <= 0:
            raise ImpossibleObservation('Release/disclosure has zero support or numerical underflow')
        if self.counters['fft_work_proxy']>0 and norm<1e-10:
            # A fail-closed numerical diagnostic, not a proved error bound. A small
            # central-mass check cannot certify all rare-posterior tail accuracy.
            raise NumericalConditioningError(f'FFT conditioning mass {norm} too small for declared numerical scope; use exact small fallback or new stable method')
        return a/norm, math.log(norm)+shift

    def posterior_target(self):
        a, _ = self._weighted_root()
        return a.sum(axis=(0, *range(2, a.ndim)))

    def risk(self, loss='zero_one'):
        p = self.posterior_target()
        if loss == 'zero_one':
            return float(1-p.max())
        if loss == 'brier':
            return float((1-np.square(p).sum())/2)
        raise ValueError(loss)

    def sample_world(self, rng):
        if self.evidence:
            raise ValueError('Final path draws require fresh unconditioned disclosure state')
        a, _ = self._weighted_root()
        flat = rng.choice(a.size, p=(a.ravel()/a.sum()))
        h, label, *count = np.unravel_index(flat, a.shape)
        result = np.empty(len(self.model.probs), dtype=int)
        def backward(node, total):
            if node.left is None:
                i = node.scope[0]
                contrib = self.contributions[i]-node.offset
                legal = np.all(contrib == total, axis=1)&self.allowed[i]
                if node.target:
                    legal &= np.asarray(self.model.target_labels) == label
                p = self.model.probs[i][h]*legal
                if p.sum() <= 0:
                    raise ArithmeticError('Backward sampled impossible leaf state')
                result[i] = rng.choice(len(p), p=p/p.sum())
                return
            l, r = node.left, node.right
            x = l.poly[h, label if l.target else 0]
            y = r.poly[h, label if r.target else 0]
            total = np.asarray(total)
            lo = np.maximum(0, total-np.asarray(y.shape)+1)
            hi = np.minimum(np.asarray(x.shape)-1, total)
            if np.any(hi < lo):
                raise ArithmeticError('Impossible count split')
            lx = tuple(slice(int(a), int(b)+1) for a, b in zip(lo, hi))
            ry = tuple(slice(int(t-b), int(t-a)+1) for t, a, b in zip(total, lo, hi))
            w = x[lx]*np.flip(y[ry])
            s = float(w.sum())
            if s <= 0:
                raise ArithmeticError('Backward split lost probability mass')
            k = rng.choice(w.size, p=w.ravel()/s)
            left_sum = lo+np.asarray(np.unravel_index(k, w.shape))
            backward(l, left_sum)
            backward(r, total-left_sum)
        backward(self.root, np.asarray(count))
        self.counters['sampled_worlds'] += 1
        return result

    @property
    def memory_bytes(self):
        return sum(n.poly.nbytes+n.logmass.nbytes+n.offset.nbytes for n in self.nodes)

    def pruning_certificate(self, owner, include_table=False):
        """Exact rank-one test of actual X_j and C_i at this prefix only."""
        from .separation import PrefixSeparation
        from .exact_tree import CertificationBudgetExceeded
        try:
            if (self._separation is None or
                    self._separation.model.identity != self.model.identity or
                    self._separation.release.identity != self.release.identity):
                self._separation = PrefixSeparation(self.model, self.release,
                                                    matrix=self.query_basis.tolist())
            return self._separation.certificate(owner, self.evidence, include_table)
        except CertificationBudgetExceeded as error:
            return {'safe_zero_current_state':False, 'owner':int(owner),
                    'evidence':dict(self.evidence), 'refusal':str(error)}

    def graph_separation_diagnostic(self, owner):
        """Conservative separation in the COMPLETE conditioned factor graph.

        Prior latent factor edges and every release scope are included. Only a
        current prefix decision is certified; no player is globally eliminated.
        """
        observed = set(self.evidence)
        remaining = set(range(len(self.model.probs)))-observed
        graph = {i:set() for i in remaining}
        graph['H'] = set()
        for i in remaining:
            p = self.model.probs[i]
            if not np.allclose(p, p[0], atol=0, rtol=0):
                graph[i].add('H'); graph['H'].add(i)
        scopes = []
        for d in range(len(self.release.observed)):
            scope = [i for i in remaining if np.ptp(self.model.contributions[i][:, d]) > 0]
            scopes.append(scope)
            # Factor node avoids forming a quadratic clique.
            f = f'Y{d}'
            graph[f] = set(scope)
            for i in scope:
                graph[i].add(f)
        seen, stack = set(), [owner]
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v); stack.extend(graph.get(v, set())-seen)
        safe = self.model.target not in seen
        return {'safe_zero_current_state':safe, 'owner':owner, 'target':self.model.target,
                'evidence':dict(self.evidence), 'release':self.release.identity,
                'model':self.model.identity, 'remaining_release_scopes':scopes,
                'owner_component':sorted(map(str,seen))}
