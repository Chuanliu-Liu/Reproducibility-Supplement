"""Exact rational sparse product tree and per-evidence risk enclosures, q<=2.

Stored binary64 prior weights/observations are interpreted as exact rationals.
All convolutions use Fraction, never FFT. Laplace likelihood exponentials are
bounded by a rational Taylor remainder with outward dyadic rounding. This
certifies the specified finite input and prefix, not a universal inference zeta,
the float sampler, or the adequacy of a real-world prior.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
from .oracle import ImpossibleObservation
from .coordinates import UnimodularCertificate, integer_inverse, transform, support_geometry


class CertificationBudgetExceeded(RuntimeError):
    pass


def exp_minus_bounds(x, bits=80):
    """Enclose exp(-x), x>=0, with explicit geometric Taylor-tail bound.

    Reduce x by powers of two to <=1. For exp(t), terms after term k+1
    have ratio <=t/(k+2), hence remainder <=next/(1-t/(k+2)). Invert
    the enclosure, square back, and round outward to a dyadic grid each time.
    """
    x = F(x)
    if x < 0 or bits < 16:
        raise ValueError('Nonnegative exponent and bits>=16 required')
    if x == 0:
        return F(1), F(1)
    denominator = 1 << bits
    def down(v):
        return F((v.numerator*denominator)//v.denominator, denominator)
    def up(v):
        return F(-((-v.numerator*denominator)//v.denominator), denominator)
    count = 0
    while x > 1:
        x /= 2
        count += 1
        if count > 64:
            raise CertificationBudgetExceeded('Exponent range reduction exceeds budget')
    term = total = F(1)
    for k in range(1, bits+32):
        term *= x/k
        total += term
        next_term = term*x/(k+1)
        remainder = next_term/(1-x/(k+2))
        if remainder < F(1, 1 << (bits+8)):
            break
    else:
        raise CertificationBudgetExceeded('Taylor remainder did not meet precision')
    lower, upper = down(1/(total+remainder)), up(1/total)
    for _ in range(count):
        lower, upper = down(lower*lower), up(upper*upper)
    return lower, upper


@dataclass
class ExactNode:
    scope: tuple
    target: bool
    left: object = None
    right: object = None
    parent: object = None
    table: object = None


class ExactProductTree:
    def __init__(self, model, release, matrix=None, cache=True,
                 max_support=100_000, max_pairs=2_000_000, audit=False):
        self.model, self.release, self.cache = model, release, cache
        self.q = len(release.observed)
        if self.q not in (1, 2) or self.q != model.contributions[0].shape[1]:
            raise ValueError('Certified supported family requires matching q<=2')
        identity = tuple(tuple(int(i == j) for j in range(self.q)) for i in range(self.q))
        self.coordinate_certificate = UnimodularCertificate.create(
            identity if matrix is None else matrix, self.q)
        self.matrix, self.inverse = self.coordinate_certificate.matrix, self.coordinate_certificate.inverse
        self.audit, self.evaluation_log = audit, []
        self.max_support, self.max_pairs = max_support, max_pairs
        self.contributions = [[transform(self.matrix, tuple(map(int, row))) for row in c]
                              for c in model.contributions]
        self.evidence, self.nodes, self.leaves = {}, [], {}
        self.counters = dict(evaluations=0, pair_products=0, updates=0, affected_nodes=0,
                             max_support=0, max_coefficient_bits=0, root_reads=0,
                             candidate_pairs=0, coordinate_additions=0, leaf_states=0)
        def build(lo, hi):
            node = ExactNode(tuple(range(lo, hi)), lo <= model.target < hi)
            if hi-lo == 1:
                self.leaves[lo] = node
            else:
                mid = (lo+hi)//2
                node.left, node.right = build(lo, mid), build(mid, hi)
                node.left.parent = node.right.parent = node
            self.nodes.append(node)
            return node
        self.root = build(0, len(model.probs))
        self._full()

    def _evaluate(self, node):
        out = {}
        pairs = candidates = leaf_states = 0
        self.counters['evaluations'] += 1
        if node.left is None:
            i = node.scope[0]
            values = [self.evidence[i]] if i in self.evidence else range(self.model.domains[i])
            for h in range(len(self.model.mixture)):
                for v in values:
                    leaf_states += 1
                    weight = F(float(self.model.probs[i][h, v]))
                    if weight:
                        label = self.model.target_labels[v] if node.target else 0
                        key = (h, label, self.contributions[i][v])
                        out[key] = out.get(key, F(0))+weight
        else:
            left, right = node.left, node.right
            candidates = len(left.table)*len(right.table)
            for (h, label_l, sum_l), a in left.table.items():
                for (g, label_r, sum_r), b in right.table.items():
                    if h != g:
                        continue
                    pairs += 1
                    if pairs > self.max_pairs:
                        raise CertificationBudgetExceeded('Exact node pair-product budget exceeded')
                    label = label_l if left.target else label_r
                    key = (h, label, tuple(x+y for x, y in zip(sum_l, sum_r)))
                    out[key] = out.get(key, F(0))+a*b
                    if len(out) > self.max_support:
                        raise CertificationBudgetExceeded('Exact node support budget exceeded')
            self.counters['pair_products'] += pairs
            self.counters['candidate_pairs'] += candidates
            self.counters['coordinate_additions'] += self.q*pairs
        self.counters['leaf_states'] += leaf_states
        node.table = out
        self.counters['max_support'] = max(self.counters['max_support'], len(out))
        if out:
            self.counters['max_coefficient_bits'] = max(self.counters['max_coefficient_bits'],
                max(max(v.numerator.bit_length(), v.denominator.bit_length()) for v in out.values()))
        if self.audit:
            channels = {}
            for h, label, sums in out:
                channels.setdefault((h, label), set()).add(sums)
            self.evaluation_log.append(dict(scope=node.scope, internal=node.left is not None,
                candidate_pairs=candidates, pair_products=pairs, rational_additions=pairs,
                coordinate_additions=self.q*pairs, leaf_states=leaf_states,
                output_entries=len(out), channels=[dict(h=h, label=a, **support_geometry(s))
                    for (h, a), s in sorted(channels.items())],
                coefficient_bits=max((max(v.numerator.bit_length(),v.denominator.bit_length())
                                      for v in out.values()), default=0)))

    def _full(self):
        for node in self.nodes:
            self._evaluate(node)

    def update(self, owner, value):
        if not 0 <= value < self.model.domains[owner]:
            raise ValueError('Bundle state out of domain')
        if self.evidence.get(owner) == value:
            return
        self.evidence[owner] = value
        dirty, node = [], self.leaves[owner]
        while node is not None:
            dirty.append(node)
            node = node.parent
        self.counters['updates'] += 1
        self.counters['affected_nodes'] += len(dirty)
        if self.cache:
            for node in dirty:
                self._evaluate(node)
        else:
            self._full()

    def reset(self):
        if self.evidence:
            self.evidence.clear()
            self._full()

    def target_joint_bounds(self, bits=80):
        self.counters['root_reads'] += 1
        count = max(self.model.target_labels)+1
        lower, upper = [F(0) for _ in range(count)], [F(0) for _ in range(count)]
        y = tuple(F(float(v)) for v in self.release.observed)
        scales = tuple(F(float(v)) for v in self.release.scales)
        entries = []
        for (h, label, sums), weight in self.root.table.items():
            original = transform(self.inverse, sums)
            weight *= F(float(self.model.mixture[h]))
            if self.release.kind == 'exact':
                if original == y:
                    lower[label] += weight
                    upper[label] += weight
            elif self.release.kind == 'laplace':
                exponent = sum(abs(F(c)-v)/s for c, v, s in zip(original, y, scales))
                entries.append((label, exponent, weight))
            else:
                raise ValueError('Certified release kind unsupported')
        if entries:
            shift = min(e for _, e, _ in entries)
            likelihoods = {}
            for label, exponent, weight in entries:
                exponent -= shift
                if exponent not in likelihoods:
                    likelihoods[exponent] = exp_minus_bounds(exponent, bits)
                lo, hi = likelihoods[exponent]
                lower[label] += weight*lo
                upper[label] += weight*hi
        if sum(upper) == 0:
            raise ImpossibleObservation('Certified zero support')
        if sum(lower) == 0:
            raise CertificationBudgetExceeded('No positive certified normalization lower bound')
        return lower, upper

    def posterior_bounds(self, bits=80):
        lower, upper = self.target_joint_bounds(bits)
        return [(max(F(0), l/sum(upper)), min(F(1), u/sum(lower))) for l, u in zip(lower, upper)]

    def risk_bounds(self, loss='zero_one', bits=80):
        p = self.posterior_bounds(bits)
        if loss == 'zero_one':
            return max(F(0), 1-max(u for _, u in p)), min(F(1), 1-max(l for l, _ in p))
        if loss == 'brier':
            return max(F(0), (1-sum(u*u for _, u in p))/2), min(F(1), (1-sum(l*l for l, _ in p))/2)
        raise ValueError(loss)

    def adaptive_risk_bounds(self, loss='zero_one', tolerance=1e-12, precisions=(48, 80, 128)):
        attempts = []
        for bits in precisions:
            lower, upper = self.risk_bounds(loss, bits)
            attempts.append(dict(bits=bits, width=float(upper-lower)))
            if upper-lower <= F(float(tolerance)):
                return lower, upper, attempts
        raise CertificationBudgetExceeded('Requested certified risk width not reached')
