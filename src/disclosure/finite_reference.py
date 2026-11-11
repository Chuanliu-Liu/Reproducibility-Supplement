"""Complete-world finite reference; no floating categorical probabilities.

Represented factors are exact rationals. Laplace exp(-x) uses the existing
independent rational Taylor enclosure. Custom likelihoods must supply certified
bounds. This route is deliberately limited to <=10 owners, not a scalable
replacement for the production sampler. HMAC seed streams are reproducibility
devices; ideal independent bits are a separate mathematical premise.
"""
from fractions import Fraction as F
import hashlib
import hmac
import itertools
import json
from .exact_tree import exp_minus_bounds, CertificationBudgetExceeded
from .oracle import ImpossibleObservation


def ratio_json(x):
    return {'numerator': str(x.numerator), 'denominator': str(x.denominator)}


class SeedBits:
    """Reproducible HMAC-SHA256 counter stream, with an exact bit cursor."""
    def __init__(self, seed, domain, position=0):
        self.seed, self.domain, self.position = str(seed), str(domain), int(position)
        self._block, self._data = -1, b''

    def bit(self):
        block, offset = divmod(self.position, 256)
        if block != self._block:
            self._data = hmac.new(self.seed.encode(),
                                  (self.domain + ':' + str(block)).encode(),
                                  hashlib.sha256).digest()
            self._block = block
        self.position += 1
        return (self._data[offset // 8] >> (7 - offset % 8)) & 1

    def uniform_integer(self, n):
        if n < 1:
            raise ValueError('Positive integer range required')
        if n == 1:
            return 0
        b = (n - 1).bit_length()
        while True:
            v = 0
            for _ in range(b):
                v = 2 * v + self.bit()
            if v < n:
                return v

    def permutation(self, players):
        out = list(players)
        for j in range(len(out)-1, 0, -1):
            k = self.uniform_integer(j+1)
            out[k], out[j] = out[j], out[k]
        return tuple(out)


def categorical_cell(cdf, word, bits):
    """Both ends must be strictly interior to the exact cumulative cell."""
    lo, hi = F(word, 1 << bits), F(word+1, 1 << bits)
    for j in range(len(cdf)-1):
        if lo > cdf[j][1] and hi < cdf[j+1][0]:
            return j
    return None


class FiniteWorldOracle:
    def __init__(self, model, release, *, likelihood_bounds=None,
                 likelihood_identity=None, max_worlds=100000):
        if len(model.domains) > 10:
            raise ValueError('Finite reference admits at most 10 owners')
        size = 1
        for d in model.domains:
            size *= d
        if size > max_worlds:
            raise CertificationBudgetExceeded('Complete world resource cap')
        if likelihood_bounds and not likelihood_identity:
            raise ValueError('Certified custom likelihood identity required')
        self.model, self.release = model, release
        self.worlds = tuple(itertools.product(*[range(d) for d in model.domains]))
        self.priors, self.counts = [], []
        for w in self.worlds:
            prior = F(0)
            for h, pi in enumerate(model.mixture):
                a = F(float(pi))
                for i, v in enumerate(w):
                    a *= F(float(model.probs[i][h, v]))
                prior += a
            self.priors.append(prior)
            self.counts.append(tuple(sum(int(model.contributions[i][v, d])
                                         for i, v in enumerate(w))
                                     for d in range(len(release.observed))))
        self.custom = likelihood_bounds
        self.identity = hashlib.sha256(json.dumps({
            'model': model.identity, 'release': release.identity,
            'likelihood': likelihood_identity, 'public_order': 'lexicographic-full-domains-v1',
            'arithmetic': 'represented-rational-priors-Taylor-exp-v1'
        }, sort_keys=True).encode()).hexdigest()
        self._mass_cache, self._cdf_cache, self._risk_cache = {}, {}, {}

    def masses(self, precision):
        if precision in self._mass_cache:
            return self._mass_cache[precision]
        cache, out = {}, []
        for w, prior, counts in zip(self.worlds, self.priors, self.counts):
            if not prior:
                out.append((F(0), F(0)))
                continue
            if counts not in cache:
                if self.custom:
                    lo, hi = self.custom(counts, precision)
                elif self.release.kind == 'exact':
                    lo = hi = F(counts == tuple(self.release.observed))
                elif self.release.kind == 'laplace':
                    exponent = sum(abs(F(c)-F(float(y)))/F(float(b))
                                   for c, y, b in zip(counts, self.release.observed,
                                                       self.release.scales))
                    # The density normalizer is a common positive constant,
                    # canceled once for the complete joint likelihood.
                    lo, hi = exp_minus_bounds(exponent, precision)
                else:
                    raise ValueError('Certified joint-likelihood adapter required')
                lo, hi = F(lo), F(hi)
                if not 0 <= lo <= hi:
                    raise ValueError('Invalid likelihood enclosure')
                cache[counts] = lo, hi
            lo, hi = cache[counts]
            out.append((prior*lo, prior*hi))
        if sum(hi for lo, hi in out) == 0:
            raise ImpossibleObservation('Exact zero normalizer')
        if sum(lo for lo, hi in out) <= 0:
            raise CertificationBudgetExceeded('Normalizer lower bound not positive; refine')
        self._mass_cache[precision] = tuple(out)
        return self._mass_cache[precision]

    def cumulative(self, precision):
        if precision in self._cdf_cache:
            return self._cdf_cache[precision]
        masses = self.masses(precision)
        total_lo, total_hi = (sum(v[k] for v in masses) for k in (0, 1))
        left_lo = left_hi = F(0)
        cdf = [(F(0), F(0))]
        for lo, hi in masses:
            left_lo += lo
            left_hi += hi
            # Monotone normalized cumulative, not midpoint probabilities.
            cdf.append((left_lo/(left_lo+total_hi-left_hi),
                        left_hi/(left_hi+total_lo-left_lo)))
        cdf[-1] = (F(1), F(1))
        self._cdf_cache[precision] = tuple(cdf)
        return self._cdf_cache[precision]

    def risk(self, evidence, loss, precision=80):
        key = tuple(sorted(evidence.items())), loss, precision
        if key in self._risk_cache:
            return self._risk_cache[key]
        groups = {}
        for w, (lo, hi) in zip(self.worlds, self.masses(precision)):
            if all(w[i] == v for i, v in evidence.items()):
                label = self.model.target_labels[w[self.model.target]]
                a, b = groups.get(label, (F(0), F(0)))
                groups[label] = a+lo, b+hi
        lower, upper = zip(*groups.values())
        if not sum(upper):
            raise ImpossibleObservation('Disclosure has exact zero support')
        if not sum(lower):
            raise CertificationBudgetExceeded('Disclosure normalization requires refinement')
        label_bounds = []
        for lo, hi in groups.values():
            label_bounds.append((lo/(lo+sum(upper)-hi) if lo else F(0),
                                 hi/(hi+sum(lower)-lo) if hi else F(0)))
        if loss == 'zero_one':
            bound = (max(F(0), 1-max(hi for lo, hi in label_bounds)),
                     min(F(1), 1-max(lo for lo, hi in label_bounds)))
        elif loss == 'brier':
            bound = (max(F(0), (1-sum(hi*hi for lo, hi in label_bounds))/2),
                     min(F(1, 2), (1-sum(lo*lo for lo, hi in label_bounds))/2))
        else:
            raise ValueError(loss)
        self._risk_cache[key] = bound
        return bound

    def path_bounds(self, world, order, loss, precision=80):
        if sorted(order) != [i for i in range(len(self.model.domains))
                             if i != self.model.target]:
            raise ValueError('Independent permutation excludes the target')
        evidence, increments = {}, {}
        prev = self.risk(evidence, loss, precision)
        for i in order:
            evidence[i] = world[i]
            risk = self.risk(evidence, loss, precision)
            increments[i] = prev[0]-risk[1], prev[1]-risk[0]
            prev = risk
        return increments

    def exact_shapley_bounds(self, loss, precision=80):
        """Enumerate all disclosure coalitions and complete worlds independently."""
        players = [i for i in range(len(self.model.domains)) if i != self.model.target]
        masses = self.masses(precision)
        zl, zu = (sum(v[k] for v in masses) for k in (0, 1))
        risks = {}
        import math
        for size in range(len(players)+1):
            for coalition in itertools.combinations(players, size):
                groups = {}
                for w, (lo, hi) in zip(self.worlds, masses):
                    e = tuple((i, w[i]) for i in coalition)
                    a, b = groups.get(e, (F(0), F(0)))
                    groups[e] = a+lo, b+hi
                rl = ru = F(0)
                for e, (lo, hi) in groups.items():
                    if not hi:
                        continue
                    risk = self.risk(dict(e), loss, precision)
                    rl += lo* risk[0]
                    ru += hi* risk[1]
                risks[coalition] = rl/zu, ru/zl
        phi = {}
        for i in players:
            lo = hi = F(0)
            for s, r in risks.items():
                if i in s:
                    continue
                nxt = risks[tuple(sorted((*s, i)))]
                weight = F(1, len(players)*math.comb(len(players)-1, len(s)))
                lo += weight*(r[0]-nxt[1])
                hi += weight*(r[1]-nxt[0])
            phi[i] = lo, hi
        return phi


class CertifiedFiniteSampler:
    def __init__(self, oracle, seed, *, precision=80, max_bits=4096,
                 max_precision=4096, checkpoint=None):
        self.oracle = oracle
        self.precision, self.max_bits, self.max_precision = precision, max_bits, max_precision
        self.world_bits = SeedBits(seed, 'world')
        self.permutation_bits = SeedBits(seed, 'permutation')
        self.completed, self.pending = 0, None
        self.max_used_precision, self.max_used_bits = 0, 0
        if checkpoint:
            if (checkpoint['oracle_identity'], checkpoint['seed']) != (oracle.identity, str(seed)):
                raise ValueError('Cannot resume a changed finite distribution or stream')
            self.world_bits.position = checkpoint['world_bit_position']
            self.permutation_bits.position = checkpoint['permutation_bit_position']
            self.completed, self.pending = checkpoint['completed'], checkpoint['pending']

    def checkpoint(self):
        return {'oracle_identity': self.oracle.identity, 'seed': self.world_bits.seed,
                'completed': self.completed, 'pending': self.pending,
                'world_bit_position': self.world_bits.position,
                'permutation_bit_position': self.permutation_bits.position,
                'ideal_iid': False, 'stream': 'HMAC-SHA256-seeded-approximation'}

    def draw(self):
        if self.pending is None:
            self.pending = {'draw_index': self.completed, 'word': 0, 'bits': 0,
                            'precision': self.precision}
        p = self.pending
        while True:
            try:
                cdf = self.oracle.cumulative(p['precision'])
                index = categorical_cell(cdf, p['word'], p['bits'])
            except CertificationBudgetExceeded:
                index = None
            if index is not None:
                world = self.oracle.worlds[index]
                players = [i for i in range(len(world)) if i != self.oracle.model.target]
                order = self.permutation_bits.permutation(players)
                self.max_used_precision = max(self.max_used_precision, p['precision'])
                self.max_used_bits = max(self.max_used_bits, p['bits'])
                self.completed += 1
                self.pending = None
                return world, order
            if p['bits'] >= self.max_bits or p['precision'] >= self.max_precision:
                raise CertificationBudgetExceeded('Pending draw retained at the same stream position')
            # Refine BOTH the probabilities and the SAME U; no redraw/discard.
            p['word'] = 2*p['word'] + self.world_bits.bit()
            p['bits'] += 1
            p['precision'] = min(self.max_precision, p['precision']+8)


def confidence_bounds(sum_lower, sum_upper, m, *, K, alpha, time_uniform=True,
                      independent_paths=True, contiguous_stream=True):
    """Outward radius via Arb; actual risk endpoint sums remain rational."""
    if K == 0:
        return {}
    if not 0 < F(alpha) < 1 or m < 1 or K < 1:
        raise ValueError('Positive m,K and alpha in (0,1) required')
    if not independent_paths or not contiguous_stream:
        raise ValueError('Coverage requires independent paths and an unskipped initial segment')
    from flint import arb, ctx
    old = ctx.prec
    try:
        ctx.prec = 128
        multiplier = m*(m+1) if time_uniform else 1
        argument = F(2*K*multiplier)/F(alpha)
        radius = ((arb(2)/m)*(arb(argument.numerator)/argument.denominator).log()).sqrt()
        man, exp = radius.upper().man_exp()
        r = F(int(man))*F(2)**int(exp)
    finally:
        ctx.prec = old
    return {i: (max(F(-1), sum_lower[i]/m-r),
                min(F(1), sum_upper[i]/m+r)) for i in sum_lower}
