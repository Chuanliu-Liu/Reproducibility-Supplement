import itertools
import math
import unittest
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np
from disclosure.model import Model, Release, latent_count_model
from disclosure.oracle import normalize_worlds, realized_risk, exact_game, ImpossibleObservation
from disclosure.compiled import Compiled, NumericalConditioningError, StateBudgetExceeded
from disclosure.estimate import path, PathBudgetExceeded
from disclosure.reverse_suffix import ReverseSuffix, SequentialFromScratch
from disclosure.reverse_suffix_exact import exact_suffix_path, rational_risk
from disclosure.reverse_suffix_controls import StoredSuffixDP


class RRCountParity:
    # Explicit joint adapter: parity is derived from the ONE retained count.
    observed, scales, kind = (0,), (1,), 'rr-count-parity'
    def __init__(self, correct=F(3, 4), observed=0):
        self.correct, self.observed = F(correct), (observed,)
        self.identity = 'rr-count-parity-'+str(correct)+'-'+str(observed)
    def rational_likelihood(self, counts):
        return self.correct if counts[0] % 2 == self.observed[0] else 1-self.correct
    def log_likelihood(self, counts):
        return np.log(np.where(np.asarray(counts)[..., 0] % 2 == self.observed[0], float(self.correct), float(1-self.correct)))


def rational_worlds(m, likelihood):
    """Independent COMPLETE-world enumeration: no convolution or RSC helper."""
    out = []
    for w in itertools.product(*[range(d) for d in m.domains]):
        prior = F(0)
        for h, pi in enumerate(m.mixture):
            weight = F(float(pi))
            for i, v in enumerate(w):
                weight *= F(float(m.probs[i][h, v]))
            prior += weight
        total = tuple(sum(int(m.contributions[i][v, d]) for i, v in enumerate(w)) for d in range(m.contributions[0].shape[1]))
        out.append((w, prior*F(likelihood(total))))
    z = sum(p for _, p in out)
    if z == 0:
        raise ImpossibleObservation('Independent world normalization is zero')
    return [(w, p/z) for w, p in out]


def rational_target(posterior, target, evidence, cardinality):
    mass = [F(0)]*cardinality
    for w, p in posterior:
        if all(w[i] == v for i, v in evidence.items()):
            mass[w[target]] += p
    if sum(mass) == 0:
        raise ImpossibleObservation('Independent evidence normalization is zero')
    return [p/sum(mass) for p in mass]


def bundle_model():
    return Model([
        [[.125, .375, .5], [.5, .25, .25]],
        [[.125, .375, .5], [.5, .25, .25]],
        [[0., 1.], [1., 0.]],
        [[1., 0.], [1., 0.]],
    ], [.25, .75], [
        [[-1, 0], [0, 1], [1, 1]],
        [[-2, 1], [-2, 1], [1, -1]],
        [[0, -1], [1, 2]],
        [[-1, 0], [2, 2]],
    ], target_labels=(0, 1, 1), version='rsc-multibundle-zeros-signed')


class ReverseSuffixTests(unittest.TestCase):
    def test_n10_exact_count_expected_shapley_from_all_worlds(self):
        m=Model([[[.5,.5]] for _ in range(10)],[1.],[[[0],[1]] for _ in range(10)],version='rsc-n10-exact-count')
        r=Release((5,),(1,),kind='exact',release_id='rsc-n10-exact-count-five')
        posterior=[(w,p) for w,p in rational_worlds(m,lambda s:F(s[0]==5)) if p]
        players=tuple(range(1,10));risks={};reference={}
        # Independent coalition oracle: group COMPLETE posterior worlds. No
        # suffix tables, production target contraction, or reference loss helper.
        for mask in range(1<<9):
            S=tuple(i for k,i in enumerate(players) if mask>>k&1);groups={}
            for w,p in posterior:
                key=tuple(w[i] for i in S)
                if key not in groups:groups[key]=[F(0),F(0)]
                groups[key][w[0]]+=p
            risks[mask]={loss:F(0) for loss in ('zero_one','brier')}
            for mass in groups.values():
                z=sum(mass);p0,p1=(v/z for v in mass)
                risks[mask]['zero_one']+=z*min(p0,p1)
                risks[mask]['brier']+=z*p0*p1
        for loss in ('zero_one','brier'):
            reference[loss]={}
            for k,i in enumerate(players):
                value=F(0)
                for mask in range(1<<9):
                    if mask>>k&1:continue
                    size=mask.bit_count();weight=F(math.factorial(size)*math.factorial(8-size),math.factorial(9))
                    value+=weight*(risks[mask][loss]-risks[mask|1<<k][loss])
                reference[loss][i]=value
        engine=ReverseSuffix(m,r,dtype='longdouble');means={loss:np.zeros(10,dtype=np.longdouble) for loss in reference}
        # For this exchangeable fixture, all complete worlds and all cyclic
        # ranks produce the exact coalition-oracle Shapley; no general cyclic
        # permutation estimator is used or claimed by the production sampler.
        for w,p in posterior:
            for shift in range(9):
                order=players[shift:]+players[:shift]
                for loss in reference:
                    out=engine.evaluate_path(w,order,loss)
                    means[loss]+=np.longdouble(p.numerator)/p.denominator*out.increments/9
        for loss in reference:
            for i in players:self.assertAlmostEqual(float(means[loss][i]),float(reference[loss][i]),places=14)
            self.assertTrue(all(v==F(1,18 if loss=='zero_one' else 36) for v in reference[loss].values()))
        print(dict(n=10,enumerated_worlds=math.prod(m.domains),positive_posterior_worlds=len(posterior),coalitions=len(risks),
                   cyclic_ranks=9,expected_shapley={loss:str(values[1]) for loss,values in reference.items()},
                   max_float_mean_discrepancy=max(abs(float(means[loss][i])-float(reference[loss][i])) for loss in reference for i in players)),flush=True)

    def test_exact_every_world_permutation_prefix_and_expected_shapley(self):
        m = bundle_model()
        # A genuinely coupled rational joint likelihood; q2 is not factorized.
        likelihood = lambda s: F(1+(s[0]+2*s[1]) % 5, 7)
        class Joint:
            observed, scales, kind, identity = (0, 0), (1, 1), 'joint-rational', 'rsc-joint-rational-v1'
            def log_likelihood(self, counts):
                c = np.asarray(counts)
                return np.log((1+(c[..., 0]+2*c[..., 1]) % 5)/7)
        release = Joint()
        posterior = rational_worlds(m, likelihood)
        float_posterior, _ = normalize_worlds(m.domains, m.log_prior, lambda w: float(release.log_likelihood(m.counts(w))))
        # The unchanged oracle forms every disclosure group, including groups
        # represented only by exact-zero worlds. Exclude ONLY these exact zeros.
        float_posterior = [(w, p) for w, p in float_posterior if p > 0]
        players = [1, 2, 3]
        for loss in ['zero_one', 'brier']:
            expected = {i: F(0) for i in players}
            engines = [ReverseSuffix(m, release, backend=b, tile_cells=3) for b in ['dense', 'sparse']]
            for w, probability in posterior:
                if not probability:
                    continue
                for order in itertools.permutations(players):
                    exact = exact_suffix_path(m, release, w, order, loss=loss, rational_likelihood=likelihood)
                    diagnostics = [engine.evaluate_path(w, order, loss, posteriors=True) for engine in engines]
                    evidence, risks = {}, []
                    for k in range(4):
                        reference = rational_target(posterior, 0, evidence, 3)
                        risk = rational_risk(reference, m.target_labels, loss)
                        self.assertEqual(exact[k]['posterior_bounds'], [(p, p) for p in reference])
                        self.assertEqual(exact[k]['risk_bounds'], (risk, risk))
                        self.assertAlmostEqual(realized_risk(float_posterior, 0, evidence, m.target_labels, loss), float(risk), places=13)
                        for diagnostic in diagnostics:
                            np.testing.assert_allclose(diagnostic.target_posteriors[k], [float(p) for p in reference], rtol=0, atol=2e-14)
                            self.assertAlmostEqual(float(diagnostic.risks[k]), float(risk), places=13)
                        risks.append(risk)
                        if k < 3:
                            evidence[order[k]] = w[order[k]]
                    for k, i in enumerate(order, 1):
                        expected[i] += probability*(risks[k-1]-risks[k])/math.factorial(3)
                        for diagnostic in diagnostics:
                            self.assertAlmostEqual(float(diagnostic.increments[i]), float(risks[k-1]-risks[k]), places=13)
                    self.assertEqual(sum(risks[k-1]-risks[k] for k in range(1, 4)), risks[0]-risks[-1])
            game = exact_game(float_posterior, 0, m.target_labels, loss)
            for i in players:
                self.assertAlmostEqual(float(expected[i]), game['phi'][i], places=13)
            self.assertGreater(engines[0].counters['zero_class_revivals'], 0)

    def test_exact_count_and_n10_oracle(self):
        for n, q, observed in [(4, 1, (2,)), (4, 2, (2, 1)), (10, 1, (4,))]:
            m = latent_count_model(n, q, seed=41)
            r = Release(observed, tuple([1]*q), kind='exact')
            posterior = rational_worlds(m, lambda s: int(s == observed))
            w = next(w for w, p in posterior if p)
            orders = [tuple(range(1, n)), tuple(range(n-1, 0, -1))]
            for order in orders:
                for loss in ['zero_one', 'brier']:
                    out = ReverseSuffix(m, r).evaluate_path(w, order, loss, posteriors=True)
                    exact = exact_suffix_path(m, r, w, order, loss=loss)
                    evidence = {}
                    for k in range(n):
                        p = rational_target(posterior, 0, evidence, 2)
                        self.assertEqual(exact[k]['posterior_bounds'], [(v, v) for v in p])
                        np.testing.assert_allclose(out.target_posteriors[k], list(map(float, p)), atol=2e-13)
                        if k < n-1:
                            evidence[order[k]] = w[order[k]]

    def test_rr_xor_joint_adapter_both_observations_noise_and_expected_shapley(self):
        m = Model([[[.5, .5]]]*3, [1.], [[[0], [1]]]*3)
        for p in [F(1, 2), F(3, 4), F(1023, 1024)]:
            for y in [0, 1]:
                r = RRCountParity(p, y)
                posterior = rational_worlds(m, r.rational_likelihood)
                expected = {1: F(0), 2: F(0)}
                for w, probability in posterior:
                    for order in [(1, 2), (2, 1)]:
                        exact = exact_suffix_path(m, r, w, order, rational_likelihood=r.rational_likelihood)
                        result = ReverseSuffix(m, r).evaluate_path(w, order)
                        self.assertEqual(exact[0]['risk_bounds'], (F(1, 2),)*2)
                        self.assertEqual(exact[1]['risk_bounds'], (F(1, 2),)*2)
                        self.assertEqual(exact[2]['risk_bounds'], (1-p,)*2)
                        for k, i in enumerate(order, 1):
                            expected[i] += probability*(exact[k-1]['risk_bounds'][0]-exact[k]['risk_bounds'][0])/2
                            self.assertAlmostEqual(float(result.increments[i]), float(exact[k-1]['risk_bounds'][0]-exact[k]['risk_bounds'][0]), places=14)
                self.assertEqual(expected, {1:(p-F(1, 2))/2, 2:(p-F(1, 2))/2})

    def test_full_original_forward_controls_and_outward_laplace_certificates(self):
        for m, r in [(bundle_model(), Release((.25, 1.5), (.75, 1.25))), (latent_count_model(5, 2, 49), Release((1.1, 2.7), (.5, .8)))]:
            posterior, _ = normalize_worlds(m.domains, m.log_prior, lambda w: float(r.log_likelihood(m.counts(w))))
            worlds = [w for w, p in posterior if p > 0][:6]
            forward = Compiled(m, r)
            for matrix in [((1, 0), (0, 1)), ((1, 0), (-1, 1))]:
                reverse = ReverseSuffix(m, r, matrix=matrix, tile_cells=4)
                scratch = SequentialFromScratch(m, r, matrix=matrix, tile_cells=4)
                suffix = StoredSuffixDP(m, r, matrix=matrix, tile_cells=4)
                for w in worlds:
                    order = tuple(range(len(w)-1, 0, -1))
                    for loss in ['zero_one', 'brier']:
                        result = reverse.evaluate_path(w, order, loss, posteriors=True)
                        reference, _ = path(forward, w, order, loss)
                        np.testing.assert_allclose(result.increments, reference, atol=2e-13, rtol=0)
                        np.testing.assert_allclose(result.increments, scratch.evaluate_path(w, order, loss).increments, atol=2e-13, rtol=0)
                        np.testing.assert_allclose(result.increments, suffix.evaluate_path(w, order, loss).increments, atol=2e-13, rtol=0)
                        exact = exact_suffix_path(m, r, w, order, loss=loss, matrix=matrix)
                        evidence = {}
                        for k in range(len(w)):
                            lo, hi = exact[k]['risk_bounds']
                            self.assertLess(float(hi-lo), 1e-24)
                            diagnostic = realized_risk(posterior, m.target, evidence, m.target_labels, loss)
                            self.assertLessEqual(max(float(lo)-diagnostic, diagnostic-float(hi), 0), 2e-13)
                            self.assertAlmostEqual(float(result.risks[k]), diagnostic, places=12)
                            if k < len(order):
                                evidence[order[k]] = w[order[k]]

    def test_hidden_suffix_and_target_value_do_not_condition_original_prefix(self):
        m, r = bundle_model(), Release((.25, 1.5), (.75, 1.25))
        engine = ReverseSuffix(m, r)
        order, w = (1, 2, 3), [0, 0, 1, 0]
        before = engine.evaluate_path(w, order, posteriors=True)
        # The zero-valued latent-class evidence is removed and its class revives.
        other = engine.evaluate_path([999, 0, 0, 0], order, posteriors=True)
        for k in [0, 1]:
            np.testing.assert_allclose(before.target_posteriors[k], other.target_posteriors[k], atol=2e-14, rtol=0)
        target_changed = engine.evaluate_path([-999, 0, 1, 0], order, posteriors=True)
        np.testing.assert_array_equal(before.risks, target_changed.risks)
        # Complete bundle evidence must distinguish states with the same f_i.
        duplicate = engine.evaluate_path([0, 1, 1, 0], order, posteriors=True)
        self.assertGreater(abs(before.risks[1]-duplicate.risks[1]), 1e-5)

    def test_fixed_hidden_nonplayers_and_both_losses(self):
        m, r = latent_count_model(5, 2, 41), Release((2.2, 1.1), (.5, .8))
        posterior, _ = normalize_worlds(m.domains, m.log_prior, lambda w: float(r.log_likelihood(m.counts(w))))
        engine = ReverseSuffix(m, r, players=[1, 3])
        for loss in ['zero_one', 'brier']:
            out = engine.evaluate_path([42, 1, 999, 0, -99], [3, 1], loss)
            evidence = {}
            for k, risk in enumerate(out.risks):
                self.assertAlmostEqual(float(risk), realized_risk(posterior, 0, evidence, m.target_labels, loss), places=13)
                if k < 2:
                    evidence[[3, 1][k]] = [0, 1][k]
        self.assertGreater(engine.counters['fixed_base_products'], 0)
        self.assertEqual(engine.retarget(2).players, (1, 3))

    def test_existing_fixed_tilt_identity_and_zero_revival(self):
        m, r = bundle_model(), Release((.25, 1.5), (.75, 1.25))
        sampler = Compiled(m, r)
        a = ReverseSuffix(m, r, matrix=sampler.query_basis.tolist(), tilt=sampler.tilt)
        b = ReverseSuffix(m, r)
        for loss in ['zero_one', 'brier']:
            np.testing.assert_allclose(a.evaluate_path([0, 0, 1, 0], [1, 2, 3], loss).risks, b.evaluate_path([2, 0, 1, 0], [1, 2, 3], loss).risks, rtol=0, atol=2e-13)
        self.assertGreater(a.counters['zero_class_revivals'], 0)

    def test_retarget_reset_new_release_and_invalid_target_permutation(self):
        m, r = latent_count_model(4, 1, 41), Release((1.2,), (.7,))
        a = ReverseSuffix(m, r)
        original = a.evaluate_path([0, 1, 1, 0], [3, 1, 2]).risks
        a.reset()
        np.testing.assert_array_equal(original, a.evaluate_path([1, 1, 1, 0], [3, 1, 2]).risks)
        b = a.retarget(2)
        posterior, _ = normalize_worlds(b.model.domains, b.model.log_prior, lambda w: float(r.log_likelihood(b.model.counts(w))))
        out = b.evaluate_path([0, 1, 999, 0], [0, 1, 3])
        self.assertAlmostEqual(float(out.risks[0]), realized_risk(posterior, 2, {}), places=13)
        changed = a.change_release(Release((3.1,), (.7,), release_id='new-rsc-release'))
        self.assertNotEqual(a.identity, changed.identity)
        self.assertGreater(abs(float(changed.evaluate_path([0, 1, 1, 0], [3, 1, 2]).risks[0]-original[0])), 1e-3)
        with self.assertRaises(ValueError):
            a.evaluate_path([0, 0, 0, 0], [0, 1, 2])

    def test_impossible_normalization_no_zero_division_and_no_support_thinning(self):
        m = Model([[[1., 0.]], [[1., 0.]]], [1.], [[[0], [1]]]*2)
        with self.assertRaises(ImpossibleObservation):
            ReverseSuffix(m, Release((1,), (1,), kind='exact')).evaluate_path([0, 0], [1])
        m = Model([[[.5, .5]], [[1e-300, 1.]], [[1e-300, 1.]]], [1.], [[[0], [1]]]*3)
        with self.assertRaises(NumericalConditioningError):
            ReverseSuffix(m, Release((1,), (1,))).evaluate_path([0, 1, 1], [1, 2])
        stable = ReverseSuffix(m, Release((1,), (1,)), dtype='longdouble')
        self.assertTrue(np.isfinite(stable.evaluate_path([0, 1, 1], [1, 2]).risks).all())
        with self.assertRaises(StateBudgetExceeded):
            ReverseSuffix(m, Release((1,), (1,)), max_coefficient_bytes=16).evaluate_path([0, 1, 1], [1, 2])

    def test_partial_checkpoint_resumes_identical_whole_path(self):
        m, r = latent_count_model(5, 2, 41), Release((2.2, .7), (.7, .9))
        engine, world, order = ReverseSuffix(m, r), [0, 1, 0, 1, 1], [3, 1, 4, 2]
        full = engine.evaluate_path(world, order, posteriors=True)
        # Cut after one reverse contraction, before a WHOLE path is counted.
        clock = iter([0., 0., 0., 0., 0., 0., 2.])
        with patch('disclosure.reverse_suffix.time.perf_counter', side_effect=lambda: next(clock, 2.)):
            with self.assertRaises(PathBudgetExceeded) as caught:
                engine.evaluate_path(world, order, deadline=1., posteriors=True)
        partial = caught.exception.partial
        resumed = engine.evaluate_path(world, order, resume=partial, posteriors=True)
        np.testing.assert_array_equal(full.risks, resumed.risks)
        bad = list(world);bad[1] = 0
        with self.assertRaises(ValueError):
            engine.evaluate_path(bad, order, resume=partial)
        with self.assertRaises(ValueError):
            ReverseSuffix(m, r, matrix=((1,0),(-1,1))).evaluate_path(world, order, resume=partial)
        with self.assertRaises(ValueError):
            ReverseSuffix(m, r, dtype='longdouble').evaluate_path(world, order, resume=partial)

    def test_negative_signed_increments_and_deliberate_wrong_algorithms(self):
        m, r = latent_count_model(4, 1, 44, strength=.3), Release((.9,), (.2,))
        posterior, _ = normalize_worlds(m.domains, m.log_prior, lambda w: float(r.log_likelihood(m.counts(w))))
        prior, _ = normalize_worlds(m.domains, m.log_prior, lambda w: 0.)
        engine = ReverseSuffix(m, r)
        flags = dict(negative=False, prior_only=False, prefix_prior=False, missing_mixture=False, target_conditioning=False, complement_sign=False)
        equal_mixture = Model(m.probs, [.5, .5], m.contributions)
        wrong = ReverseSuffix(equal_mixture, r)
        for w, probability in posterior:
            order = (1, 2, 3)
            out = engine.evaluate_path(w, order)
            flags['negative'] |= bool((out.increments < -1e-10).any())
            flags['prior_only'] |= abs(float(out.risks[0])-realized_risk(prior, 0, {})) > 1e-5
            flags['prefix_prior'] |= abs(float(out.risks[1])-realized_risk(prior, 0, {1:w[1]})) > 1e-5
            flags['missing_mixture'] |= bool(np.max(abs(out.risks-wrong.evaluate_path(w, order).risks)) > 1e-5)
            flags['target_conditioning'] |= out.risks[0] > 1e-5  # Fixing Xj would give zero.
            complement = {i:w[i] for i in order[1:]}
            wrong_delta = realized_risk(posterior, 0, {i:w[i] for i in order})-realized_risk(posterior, 0, complement)
            flags['complement_sign'] |= abs(float(out.increments[1])-wrong_delta) > 1e-5
        self.assertEqual(flags, {k:True for k in flags})


if __name__ == '__main__':
    unittest.main()
