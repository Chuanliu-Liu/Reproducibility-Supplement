"""Exact, prefix-specific release-conditioned rank-one certificates.

The stored binary64 inputs are interpreted as rationals, as in ExactProductTree.
A Laplace cell is a finite positive sum a_r exp(-r), r rational. Minors are
multiplied and cancelled symbolically; no tolerance, MI estimate, or rounded
posterior enters a certificate. With a common denominator L these expressions
are polynomials in exp(-1/L), a transcendental number. Thus the canonical
coefficient test is exact. Resource exhaustion refuses the skip.
"""
from dataclasses import replace
from fractions import Fraction as F
import hashlib
import json
import time

from .exact_tree import ExactProductTree, CertificationBudgetExceeded
from .coordinates import transform
from .oracle import ImpossibleObservation


def loss_sufficient(statistic, loss_rows):
    """Check identical loss vectors on every statistic fiber, for ALL actions.

    This is a sufficient deterministic condition, not a test of query-answer
    independence. Rows are actions, columns are actual target owner states.
    """
    statistic = tuple(statistic)
    rows = [tuple(F(v) for v in row) for row in loss_rows]
    if not rows or any(len(row) != len(statistic) for row in rows):
        raise ValueError('Every declared action must specify all target-state losses')
    if any(v < 0 or v > 1 for row in rows for v in row):
        raise ValueError('Declared loss outside [0,1]')
    return all(all(statistic[x] != statistic[y] or row[x] == row[y]
                   for x in range(len(statistic)) for y in range(x)) for row in rows)


def serialize_table(table):
    return [[[ [str(r), str(a)] for r, a in sorted(cell.items())]
             for cell in row] for row in table]


def rank_one(table, counters=None, max_products=2_000_000):
    """Exact pivot-minor check, including holes in positive marginal support."""
    counters = {} if counters is None else counters
    if not table or not table[0] or any(len(row) != len(table[0]) for row in table):
        raise ValueError('Rectangular joint table required')
    if any(a <= 0 for row in table for cell in row for a in cell.values()):
        raise ValueError('Joint-table coefficients must be positive')
    rows = [x for x, row in enumerate(table) if any(row)]
    cols = [c for c in range(len(table[0])) if any(row[c] for row in table)]
    if not rows or not cols:
        raise ImpossibleObservation('Exact joint table has zero support')
    # Rank one needs the entire Cartesian product of positive marginals.
    # Two disconnected diagonal cells are dependence, not "rank one per block".
    for x in rows:
        for c in cols:
            if not table[x][c]:
                return False, {'kind':'support_hole', 'cell':[x,c]}
    x0, c0 = rows[0], cols[0]
    products = 0
    for x in rows[1:]:
        for c in cols[1:]:
            counters['minors_checked'] = counters.get('minors_checked', 0)+1
            minor = {}
            for left, right, sign in [(table[x0][c0], table[x][c], 1),
                                      (table[x0][c], table[x][c0], -1)]:
                for r, a in left.items():
                    for s, b in right.items():
                        products += 1
                        counters['minor_term_products'] = counters.get('minor_term_products', 0)+1
                        if products > max_products:
                            raise CertificationBudgetExceeded('Exact minor product budget exceeded')
                        exponent = r+s
                        minor[exponent] = minor.get(exponent, F(0))+sign*a*b
            residual = {r:a for r,a in minor.items() if a}
            if residual:
                r = min(residual)
                return False, {'kind':'nonzero_minor', 'rows':[x0,x], 'cols':[c0,c],
                               'first_exponent':str(r), 'first_coefficient':str(residual[r]),
                               'residual_terms':len(residual)}
    return True, None


class PrefixSeparation:
    """Actual X_j channel by default; optional loss-sufficient statistic only.

    The cavity joins unchanged sibling tables along the candidate's ancestor
    path. It retains H until the full release likelihood has been applied and
    H has been summed out. Each call reconstructs the joint for its exact e.
    """
    def __init__(self, model, release, matrix=None, statistic=None, loss_rows=None,
                 max_support=100_000, max_pairs=2_000_000):
        self.model, self.release = model, release
        self.statistic = tuple(range(model.domains[model.target])) if statistic is None else tuple(statistic)
        if len(self.statistic) != model.domains[model.target] or any(
                not isinstance(x, int) or x < 0 for x in self.statistic):
            raise ValueError('Invalid finite target statistic')
        self.sufficiency = 'full_actual_target_state'
        if statistic is not None:
            if loss_rows is None or not loss_sufficient(self.statistic, loss_rows):
                raise ValueError('Target statistic lacks declared loss sufficiency')
            self.sufficiency = 'identical_declared_action_loss_vectors_on_fibers'
        self.loss_rows = loss_rows
        self.max_support, self.max_pairs = max_support, max_pairs
        self.tree = ExactProductTree(replace(model, target_labels=self.statistic), release,
                                     matrix=matrix, max_support=max_support, max_pairs=max_pairs)
        self.counters = dict(calls=0, certified=0, refused=0, cavity_pair_products=0,
                             joint_terms=0, minors_checked=0, minor_term_products=0)
        self.seconds = 0.
        self.refused_reason = None

    def synchronize(self, evidence):
        if self.refused_reason is not None:
            raise CertificationBudgetExceeded(self.refused_reason)
        if any(evidence.get(i) != v for i,v in self.tree.evidence.items()):
            self.tree.reset()
        for i,v in evidence.items():
            if i == self.model.target:
                raise ValueError('Target excluded from disclosure prefix')
            try:
                self.tree.update(int(i), int(v))
            except CertificationBudgetExceeded as error:
                # ExactProductTree may have changed evidence before exhausting
                # a node budget. Never reuse possibly partial messages.
                self.refused_reason = str(error)
                raise

    def joint_table(self, owner, evidence):
        if owner == self.model.target or owner in evidence or owner not in self.tree.leaves:
            raise ValueError('Candidate must be an undisclosed non-target owner')
        self.synchronize(evidence)
        tree = self.tree
        cavity = {(h, 0, (0,)*tree.q):F(1) for h in range(len(self.model.mixture))}
        node = tree.leaves[owner]
        has_target = False
        while node.parent is not None:
            sibling = node.parent.right if node is node.parent.left else node.parent.left
            out = {}; pairs = 0
            for (h,x,s),a in cavity.items():
                for (g,label,t),b in sibling.table.items():
                    if h != g:
                        continue
                    pairs += 1
                    self.counters['cavity_pair_products'] += 1
                    if pairs > self.max_pairs:
                        raise CertificationBudgetExceeded('Exact cavity pair budget exceeded')
                    key = (h, x if has_target else label, tuple(u+v for u,v in zip(s,t)))
                    out[key] = out.get(key,F(0))+a*b
                    if len(out) > self.max_support:
                        raise CertificationBudgetExceeded('Exact cavity support budget exceeded')
            cavity = out
            has_target |= sibling.target
            node = node.parent
        table = [[{} for _ in range(self.model.domains[owner])]
                 for _ in range(max(self.statistic)+1)]
        y = tuple(F(float(v)) for v in self.release.observed)
        scales = tuple(F(float(v)) for v in self.release.scales)
        for (h,x,s),a in cavity.items():
            for c,t in enumerate(tree.contributions[owner]):
                weight = a*F(float(self.model.mixture[h]))*F(float(self.model.probs[owner][h,c]))
                if not weight:
                    continue
                original = transform(tree.inverse, tuple(u+v for u,v in zip(s,t)))
                if self.release.kind == 'exact':
                    if original != y:
                        continue
                    exponent = F(0)
                elif self.release.kind == 'laplace':
                    exponent = sum(abs(F(v)-o)/scale for v,o,scale in zip(original,y,scales))
                else:
                    raise ValueError('Exact certificate supports exact/Laplace releases only')
                cell = table[x][c]
                cell[exponent] = cell.get(exponent,F(0))+weight
                self.counters['joint_terms'] += 1
        # Remove one common positive scale only, never a row/column-specific one.
        exponents = [r for row in table for cell in row for r in cell]
        if not exponents:
            raise ImpossibleObservation('Release/prefix has zero exact support')
        shift = min(exponents)
        return [[{r-shift:a for r,a in cell.items()} for cell in row] for row in table], shift

    def certificate(self, owner, evidence, include_table=False):
        started = time.perf_counter(); self.counters['calls'] += 1
        result = dict(safe_zero_current_state=False, owner=int(owner), target=self.model.target,
                      evidence={int(i):int(v) for i,v in evidence.items()},
                      release=self.release.identity, model=self.model.identity,
                      rule='exact_release_conditioned_joint_rank_one', prefix_specific=True,
                      target_statistic=list(self.statistic), sufficiency=self.sufficiency,
                      arithmetic='rational coefficients and rational-exponent exp polynomials')
        try:
            table, shift = self.joint_table(owner, evidence)
            safe, witness = rank_one(table, self.counters, self.max_pairs)
            encoded = serialize_table(table)
            result.update(safe_zero_current_state=safe, witness=witness,
                          table_sha256=hashlib.sha256(json.dumps(encoded,separators=(',',':')).encode()).hexdigest(),
                          removed_common_exponent=str(shift),
                          positive_rows=sum(any(row) for row in table),
                          positive_columns=sum(any(row[c] for row in table) for c in range(len(table[0]))),
                          supported_values=[c for c in range(len(table[0])) if any(row[c] for row in table)])
            if include_table:
                result['joint_table'] = encoded
            self.counters['certified'] += int(safe)
        except CertificationBudgetExceeded as error:
            self.counters['refused'] += 1
            self.refused_reason = str(error)
            result['refusal'] = str(error)
        finally:
            self.seconds += time.perf_counter()-started
        return result
