import itertools
import unittest
from fractions import Fraction as F
import numpy as np
from disclosure.coordinates import UnimodularCertificate, transform, support_geometry
from disclosure.exact_tree import ExactProductTree
from disclosure.compiled import Compiled
from disclosure.model import Model, Release, latent_count_model
from disclosure.oracle import ImpossibleObservation
from disclosure.baselines import Enumerated


MATRICES = [((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 2), (1, 3)),
            ((-1, 0), (0, 1)), ((1, -1), (0, 1)), ((1, 0), (-1, 1))]


def signed_model():
    return Model([np.array([[.25, .25, .5], [.5, .25, .25]]),
                  np.array([[.25, .75], [.5, .5]]),
                  np.array([[.5, .5], [.75, .25]])], np.array([.5, .5]),
                 [np.array([[0, -1], [1, 0], [2, 2]]),
                  np.array([[-2, 1], [0, 0]]), np.array([[1, 1], [3, 0]])],
                 target_labels=(0, 1, 2))


def rational_masses(model, answer, evidence):
    masses = [F(0)]*(max(model.target_labels)+1)
    for world in itertools.product(*(range(d) for d in model.domains)):
        if any(world[i] != v for i, v in evidence.items()):
            continue
        if tuple(model.counts(world)) != tuple(answer):
            continue
        for h in range(len(model.mixture)):
            mass = F(float(model.mixture[h]))
            for i, v in enumerate(world):
                mass *= F(float(model.probs[i][h, v]))
            masses[model.target_labels[world[model.target]]] += mass
    return masses


class R3Certificates(unittest.TestCase):
    def test_public_general_matrix_float_direct_matches_independent_oracle(self):
        model=signed_model()
        release=Release((.75,-.25),(.5,1.25))
        reference=Enumerated(model,release)
        for matrix in MATRICES:
            engine=Compiled(model,release,query_matrix=matrix,convolution_method='direct')
            for states in itertools.product([-1,0,1],repeat=2):
                engine.reset();reference.reset()
                for i,v in enumerate(states,1):
                    if v>=0:engine.update(i,v);reference.update(i,v)
                np.testing.assert_allclose(engine.posterior_target(),reference.posterior_target(),atol=1e-12,rtol=0)

    def test_official_sage_exhaustive_expected_same_game(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from compare_baselines_r3 import FullCoalitionImputer, ReplayPermutations, PermutationEstimator, utils
        from disclosure.oracle import normalize_worlds,exact_game
        model=latent_count_model(3,2,41)
        release=Release((1.2,.6),(.75,.5))
        posterior,_=normalize_worlds(model.domains,model.log_prior,lambda w:float(release.log_likelihood(model.counts(w))))
        for loss in ['zero_one','brier']:
            imputer=FullCoalitionImputer(model,release,loss)
            official=PermutationEstimator(imputer,'mse',n_jobs=1,random_state=19)
            if loss=='zero_one':official.loss_fn=utils.ZeroOneLoss(reduction='none')
            weighted=np.zeros(2)
            orders=list(itertools.permutations(range(2)))
            for world,probability in posterior:
                official.rng=ReplayPermutations(orders)
                X=np.tile(np.asarray(world)[1:],(len(orders),1))
                Y=np.full(len(orders),world[0])
                scores,_=official._process_sample(X,Y,2,0,2)
                weighted+=probability*scores.mean(axis=0)
            phi=exact_game(posterior,0,loss=loss)['phi']
            np.testing.assert_allclose(weighted,[phi[1],phi[2]],atol=1e-12,rtol=0)

    def test_fail_closed_integer_dimension_and_determinant(self):
        for matrix in [(), ((1, 0),), ((1, 0), (0, 0)), ((2, 0), (0, 1)),
                       ((1., 0), (0, 1)), ((True,),), ((.5,),),
                       ((1, 0, 0), (0, 1, 0), (0, 0, 1))]:
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                UnimodularCertificate.create(matrix)
        for q, matrix in [(1, MATRICES[0]), (2, ((1,),))]:
            model = latent_count_model(3, q)
            release = Release(tuple([1.]*q), tuple([1.]*q))
            with self.assertRaises(ValueError):
                ExactProductTree(model, release, matrix=matrix)
            with self.assertRaises(ValueError):
                Compiled(model, release, query_matrix=matrix)
        with self.assertRaises(ValueError):
            transform(MATRICES[0], (1,))
        # Exact inverse construction must not go through binary64 linear algebra.
        k = 2**60
        cert = UnimodularCertificate.create(((1, k), (0, 1)))
        self.assertEqual(cert.decode(cert.encode((3, -7))), (3, -7))

    def test_exhaustive_exact_posterior_risk_and_laplace_invariance(self):
        model = signed_model()
        worlds = list(itertools.product(*(range(d) for d in model.domains)))
        support = {tuple(model.counts(w)) for w in worlds}
        for matrix in MATRICES:
            cert = UnimodularCertificate.create(matrix, 2)
            encoded = {cert.encode(s) for s in support}
            self.assertEqual({cert.decode(u) for u in encoded}, support)
            self.assertEqual(len(encoded), len(support))
            for observed in sorted(support):
                release = Release(observed, (1, 1), kind='exact')
                engine = ExactProductTree(model, release, matrix=matrix)
                for states in itertools.product([-1, 0, 1], repeat=2):
                    evidence = {i:v for i,v in enumerate(states, 1) if v >= 0}
                    engine.reset()
                    for i, v in evidence.items():
                        engine.update(i, v)
                    masses = rational_masses(model, observed, evidence)
                    if not sum(masses):
                        with self.assertRaises(ImpossibleObservation):
                            engine.posterior_bounds()
                    else:
                        p = [m/sum(masses) for m in masses]
                        self.assertEqual(engine.posterior_bounds(), [(a, a) for a in p])
                        zero_one = 1-max(p)
                        brier = (1-sum(a*a for a in p))/2
                        self.assertEqual(engine.risk_bounds(), (zero_one, zero_one))
                        self.assertEqual(engine.risk_bounds('brier'), (brier, brier))
            release = Release((.75, -.25), (.5, 1.25))
            original = ExactProductTree(model, release)
            mapped = ExactProductTree(model, release, matrix=matrix)
            for states in itertools.product([-1, 0, 1], repeat=2):
                original.reset(); mapped.reset()
                for i, v in enumerate(states, 1):
                    if v >= 0:
                        original.update(i, v); mapped.update(i, v)
                decoded = {(h,a,cert.decode(u)):v for (h,a,u),v in mapped.root.table.items()}
                self.assertEqual(decoded, original.root.table)
                self.assertEqual(mapped.posterior_bounds(), original.posterior_bounds())
                for loss in ['zero_one', 'brier']:
                    self.assertEqual(mapped.adaptive_risk_bounds(loss)[:2],
                                     original.adaptive_risk_bounds(loss)[:2])

    def test_q1_sign_bijection_all_worlds_and_disclosures(self):
        model = latent_count_model(3, 1, 41)
        for world in itertools.product([0, 1], repeat=3):
            release = Release(tuple(map(int, model.counts(world))), (1,), kind='exact')
            positive = ExactProductTree(model, release)
            negative = ExactProductTree(model, release, matrix=((-1,),))
            for states in itertools.product([-1, 0, 1], repeat=2):
                positive.reset(); negative.reset()
                for i, v in enumerate(states, 1):
                    if v >= 0:
                        positive.update(i, v);negative.update(i, v)
                masses = rational_masses(model, release.observed, positive.evidence)
                if sum(masses):
                    self.assertEqual(positive.posterior_bounds(), negative.posterior_bounds())
                else:
                    with self.assertRaises(ImpossibleObservation):negative.posterior_bounds()

    def test_cost_identity_every_world_permutation_and_offpath_immutability(self):
        for n in [2, 4]:
            model = latent_count_model(n, 2, 531)
            release = Release((1., 1.), (1., 1.))
            cached = ExactProductTree(model, release, audit=True)
            full = ExactProductTree(model, release, cache=False, audit=True)
            for world in itertools.product([0, 1], repeat=n):
                for order in itertools.permutations(range(1, n)):
                    cached.reset();full.reset()
                    cached.evaluation_log.clear();full.evaluation_log.clear()
                    for i in order:
                        before = {v.scope:v.table for v in cached.nodes}
                        offset_c, offset_f = len(cached.evaluation_log), len(full.evaluation_log)
                        cached.update(i, world[i]);full.update(i, world[i])
                        c = cached.evaluation_log[offset_c:];f = full.evaluation_log[offset_f:]
                        dirty = {tuple(v['scope']) for v in c}
                        for node, fresh in zip(cached.nodes, full.nodes):
                            self.assertEqual(node.table, fresh.table)
                            if node.scope not in dirty:self.assertIs(node.table, before[node.scope])
                        costs = {tuple(v['scope']):v for v in f}
                        for row in c:self.assertEqual(row, costs[tuple(row['scope'])])
                        omitted = [v for v in f if v['internal'] and tuple(v['scope']) not in dirty]
                        path_cost = sum(v['pair_products'] for v in c)
                        full_cost = sum(v['pair_products'] for v in f)
                        self.assertEqual(full_cost-path_cost, sum(v['pair_products'] for v in omitted))
                        self.assertEqual(full_cost > path_cost, bool(omitted))
                    before_count = cached.counters['evaluations']
                    cached.update(order[-1], world[order[-1]])
                    self.assertEqual(cached.counters['evaluations'], before_count)
                    cached.evaluation_log.clear();full.evaluation_log.clear()

    def test_geometry_tie_worsening_and_wrong_likelihood_counterexample(self):
        support = set(itertools.product(range(3), repeat=2))
        original = support_geometry(support)
        swap = UnimodularCertificate.create(((0, 1), (1, 0)))
        shear = UnimodularCertificate.create(((1, 1), (0, 1)))
        self.assertEqual(support_geometry(map(swap.encode, support))['bounding_volume'], 9)
        worse = support_geometry(map(shear.encode, support))
        self.assertEqual(worse['cardinality'], original['cardinality'])
        self.assertEqual(worse['bounding_volume'], 15)
        # Correct original-coordinate likelihoods agree; independent transformed
        # Laplace factors change the posterior even under a unimodular shear.
        cert = UnimodularCertificate.create(((1, 0), (-1, 1)))
        release = Release((0., 0.), (1., 1.))
        worlds = [(1, 0), (0, 1)]
        correct = np.exp([release.log_likelihood(s) for s in worlds])
        wrong = np.exp([release.log_likelihood(cert.encode(s)) for s in worlds])
        self.assertAlmostEqual(correct[0]/sum(correct), .5)
        self.assertGreater(abs(wrong[0]/sum(wrong)-.5), .2)


if __name__ == '__main__':unittest.main()
