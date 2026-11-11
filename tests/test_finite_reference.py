import itertools
import json
import unittest
from fractions import Fraction as F
import numpy as np
from disclosure.model import Model, Release
from disclosure.finite_reference import (FiniteWorldOracle, CertifiedFiniteSampler,
                                         SeedBits, categorical_cell, confidence_bounds)
from disclosure.exact_tree import CertificationBudgetExceeded
from test_reverse_suffix import bundle_model, rational_worlds, rational_target


class FixedBits:
    def __init__(self, values):
        self.values, self.position = values, 0
    def bit(self):
        v = self.values[self.position]
        self.position += 1
        return v


class FiniteReferenceTests(unittest.TestCase):
    def binary(self, n=3):
        return Model([[[.5, .5]] for _ in range(n)], [1.],
                     [[[0], [1]] for _ in range(n)], version='finite-test')

    def test_exact_dyadic_cells_strict_boundary(self):
        cdf = [(F(0), F(0)), (F(1, 2), F(1, 2)), (F(1), F(1))]
        self.assertIsNone(categorical_cell(cdf, 1, 2))  # U includes 1/2.
        self.assertEqual(categorical_cell(cdf, 2, 3), 0)
        self.assertEqual(categorical_cell(cdf, 5, 3), 1)
        self.assertIsNone(categorical_cell(cdf, 0, 3))

    def test_same_boundary_U_refines_without_redraw(self):
        m = self.binary(1)
        o = FiniteWorldOracle(m, Release((0,), (1,), kind='exact'),
                             likelihood_bounds=lambda s, b: (F(1), F(1)),
                             likelihood_identity='equal')
        sampler = CertifiedFiniteSampler(o, 1)
        bits = FixedBits([0, 1, 1, 0])
        sampler.world_bits = bits
        w, order = sampler.draw()
        self.assertEqual((w, order), ((0,), ()))
        self.assertEqual(bits.position, 4)
        self.assertEqual(sampler.completed, 1)

    def test_cutoff_checkpoint_resume_matches_uncapped_stream(self):
        m = self.binary()
        o = FiniteWorldOracle(m, Release((1,), (1,), kind='exact'))
        capped = CertifiedFiniteSampler(o, 'resume-seed', max_bits=1)
        with self.assertRaises(CertificationBudgetExceeded):
            capped.draw()
        saved = json.loads(json.dumps(capped.checkpoint()))
        self.assertEqual(saved['completed'], 0)
        self.assertEqual(saved['pending']['draw_index'], 0)
        resumed = CertifiedFiniteSampler(o, 'resume-seed', checkpoint=saved)
        direct = CertifiedFiniteSampler(o, 'resume-seed')
        for _ in range(50):
            self.assertEqual(resumed.draw(), direct.draw())
        self.assertEqual(resumed.world_bits.position, direct.world_bits.position)
        self.assertEqual(resumed.permutation_bits.position, direct.permutation_bits.position)
        with self.assertRaises(ValueError):
            CertifiedFiniteSampler(o, 'changed', checkpoint=saved)

    def test_structural_zero_and_n10_complete_support(self):
        m = self.binary(10)
        o = FiniteWorldOracle(m, Release((5,), (1,), kind='exact'))
        masses = o.masses(80)
        self.assertEqual(len(masses), 1024)
        self.assertEqual(sum(lo > 0 for lo, hi in masses), 252)
        s = CertifiedFiniteSampler(o, 81)
        for _ in range(20):
            w, order = s.draw()
            self.assertEqual(sum(w), 5)
            self.assertEqual(sorted(order), list(range(1, 10)))

    def test_nonuniform_transcendental_enclosures_and_risks(self):
        from flint import arb, ctx
        m = self.binary(3)
        r = Release((.7,), (.3,))
        o = FiniteWorldOracle(m, r)
        old = ctx.prec
        try:
            ctx.prec = 256
            values = []
            for w, prior, counts, (lo, hi) in zip(o.worlds, o.priors, o.counts, o.masses(96)):
                e = abs(F(counts[0])-F(.7))/F(.3)
                v = (-(arb(e.numerator)/e.denominator)).exp()*arb(prior.numerator)/prior.denominator
                self.assertTrue(v >= arb(lo.numerator)/lo.denominator)
                self.assertTrue(v <= arb(hi.numerator)/hi.denominator)
                values.append(v)
            z = sum(values)
            for k, (lo, hi) in enumerate(o.cumulative(96)):
                if k in (0, len(values)):
                    self.assertEqual((lo, hi), (F(k == len(values)),)*2)
                    continue  # Exact endpoint identity; independent Arb division
                              # encloses 1 with radius and cannot prove >=1.
                v = sum(values[:k])/z if k else arb(0)
                self.assertTrue(v >= arb(lo.numerator)/lo.denominator)
                self.assertTrue(v <= arb(hi.numerator)/hi.denominator)
        finally:
            ctx.prec = old
        for loss in ['zero_one', 'brier']:
            for w in o.worlds:
                for order in itertools.permutations([1, 2]):
                    inc = o.path_bounds(w, order, loss, 96)
                    self.assertTrue(all(-1 <= lo <= hi <= 1 for lo, hi in inc.values()))

    def test_zero_class_revival_full_bundle_signed_losses(self):
        m = bundle_model()
        lk = lambda s: F(1+(s[0]+2*s[1])%5, 7)
        o = FiniteWorldOracle(m, Release((0, 0), (1, 1)),
                             likelihood_bounds=lambda s, b: (lk(s), lk(s)),
                             likelihood_identity='original-multistate-joint-v1')
        complete = rational_worlds(m, lk)
        negative = 0
        for w, p in complete:
            if not p:
                continue
            for order in itertools.permutations([1, 2, 3]):
                for loss in ['zero_one', 'brier']:
                    e = {}
                    for i in order:
                        posterior = rational_target(complete, 0, e, 3)
                        label = [posterior[0], posterior[1]+posterior[2]]
                        exact = 1-max(label) if loss == 'zero_one' else (1-sum(x*x for x in label))/2
                        self.assertEqual(o.risk(e, loss), (exact, exact))
                        e[i] = w[i]
                    negative += sum(lo < 0 for lo, hi in o.path_bounds(w, order, loss).values())
        self.assertGreater(negative, 0)

    def test_RR_XOR_exact_shapley(self):
        m = self.binary()
        o = FiniteWorldOracle(m, Release((0,), (1,)),
                             likelihood_bounds=lambda s, b: ((F(3,4),)*2 if s[0]%2 == 0 else (F(1,4),)*2),
                             likelihood_identity='original-RR-count-parity-3/4')
        self.assertEqual(o.exact_shapley_bounds('zero_one'), {1: (F(1,8), F(1,8)), 2: (F(1,8), F(1,8))})

    def test_exact_uniform_integer_rejection_and_permutations(self):
        b = SeedBits(1, 'test')
        values = FixedBits([1, 1, 0, 1])
        b.bit = values.bit
        self.assertEqual(b.uniform_integer(3), 1)  # Reject binary11; accept01.
        self.assertEqual(values.position, 4)
        b = SeedBits(17, 'perm')
        counts = {p: 0 for p in itertools.permutations([1, 2, 3])}
        for _ in range(6000):
            counts[b.permutation([1, 2, 3])] += 1
        self.assertTrue(all(800 < v < 1200 for v in counts.values()))

    def test_dependence_and_skipping_invalidate_coverage(self):
        for kw in [{'independent_paths': False}, {'contiguous_stream': False}]:
            with self.assertRaises(ValueError):
                confidence_bounds({1: F(0)}, {1: F(0)}, 10, K=1, alpha=F(1,20), **kw)
        self.assertEqual(confidence_bounds({}, {}, 0, K=0, alpha=F(1,20)), {})
        fixed = confidence_bounds({1: F(0)}, {1: F(0)}, 512, K=1, alpha=F(1,20), time_uniform=False)
        uniform = confidence_bounds({1: F(0)}, {1: F(0)}, 512, K=1, alpha=F(1,20))
        self.assertLess(uniform[1][0], fixed[1][0])
        self.assertGreater(uniform[1][1], fixed[1][1])


if __name__ == '__main__':
    unittest.main()
