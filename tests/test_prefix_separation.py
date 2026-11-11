import itertools
import unittest
from unittest.mock import patch
from fractions import Fraction as F
import numpy as np

from disclosure.model import Model, Release, latent_count_model
from disclosure.compiled import Compiled
from disclosure.separation import PrefixSeparation, rank_one, loss_sufficient
from disclosure.exact_tree import CertificationBudgetExceeded
from disclosure.oracle import ImpossibleObservation
from disclosure.baselines import Enumerated
from disclosure.estimate import path


def enumeration_joint(model, release, evidence, owner):
    """Independent world enumeration; no tree messages or cavity computation."""
    table = [[{} for _ in range(model.domains[owner])]
             for _ in range(model.domains[model.target])]
    for world in itertools.product(*(range(d) for d in model.domains)):
        if any(world[i] != v for i,v in evidence.items()):
            continue
        weight = sum(F(float(model.mixture[h]))*np.prod(
            [F(float(p[h,v])) for p,v in zip(model.probs,world)])
            for h in range(len(model.mixture)))
        if not weight:
            continue
        counts = tuple(map(int, model.counts(world)))
        y = tuple(F(float(v)) for v in release.observed)
        if release.kind == 'exact':
            if counts != y:
                continue
            r = F(0)
        else:
            r = sum(abs(F(c)-v)/F(float(s)) for c,v,s in zip(counts,y,release.scales))
        cell = table[world[model.target]][world[owner]]
        cell[r] = cell.get(r,F(0))+weight
    exponents = [r for row in table for cell in row for r in cell]
    if not exponents:
        raise ImpossibleObservation('Zero enumerated support')
    shift = min(exponents)
    return [[{r-shift:a for r,a in cell.items()} for cell in row] for row in table],shift


def xor_table(evidence, owner, p=F(3,4)):
    table = [[{} for _ in range(2)] for _ in range(2)]
    for world in itertools.product(range(2), repeat=3):
        if any(world[i] != v for i,v in evidence.items()):
            continue
        a = (p if (world[0]^world[1]^world[2]) == 1 else 1-p)/8
        cell = table[world[0]][world[owner]]
        cell[F(0)] = cell.get(F(0),F(0))+a
    return table


def exact_table_risk(table, column=None):
    mass = [sum((cell.get(F(0), F(0)) for c,cell in enumerate(row)
                 if column is None or c == column),F(0)) for row in table]
    return 1-max(mass)/sum(mass)


def statistic_counterexample():
    # X0=(A,B), C1=B, query answer=A. H=B is an exact finite mixture.
    return Model([np.array([[.5,0,.5,0],[0,.5,0,.5]]),np.array([[1.,0],[0,1.]])],
                 np.array([.5,.5]),[np.array([[0],[0],[1],[1]]),np.array([[0],[0]])],
                 target_labels=(0,1,0,1))


class PrefixCertificates(unittest.TestCase):
    def test_exact_symbolic_joint_matches_all_worlds_prefixes_and_coordinates(self):
        model = Model([np.array([[.25,.25,.5],[.5,.25,.25]]),
                       np.array([[.25,.75],[.5,.5]]),np.array([[.5,.5],[.75,.25]])],
                      np.array([.5,.5]),[np.array([[0,-1],[1,0],[2,2]]),
                      np.array([[-2,1],[0,0]]),np.array([[1,1],[3,0]])],
                      target_labels=(0,1,1))  # Default certificate must retain all 3 X states.
        answers = sorted({tuple(map(int,model.counts(w))) for w in itertools.product(range(3),range(2),range(2))})
        releases = [Release(y,(1,1),kind='exact') for y in answers]+[Release((.75,-.25),(.5,1.25))]
        for release in releases:
            for matrix in [None,((1,0),(-1,1)),((0,1),(1,0))]:
                certifier = PrefixSeparation(model,release,matrix=matrix)
                for owner in [1,2]:
                    other = 3-owner
                    for state in [-1,0,1]:
                        evidence = {} if state < 0 else {other:state}
                        try: expected,shift = enumeration_joint(model,release,evidence,owner)
                        except ImpossibleObservation:
                            with self.assertRaises(ImpossibleObservation):certifier.joint_table(owner,evidence)
                            continue
                        self.assertEqual(certifier.joint_table(owner,evidence),(expected,shift))
                        accepted = certifier.certificate(owner,evidence)['safe_zero_current_state']
                        self.assertEqual(accepted,rank_one(expected)[0])
                        if accepted:
                            reference = Enumerated(model,release)
                            reference.evidence = dict(evidence)
                            for loss in ['zero_one','brier']:
                                before = reference.risk(loss)
                                for c in range(model.domains[owner]):
                                    if not any(row[c] for row in expected):continue
                                    reference.evidence = {**evidence,owner:c}
                                    self.assertAlmostEqual(reference.risk(loss),before,places=12)
                                reference.evidence = dict(evidence)

    def test_rr_xor_prefix_synergy_and_pairwise_mi_counterexample(self):
        p = F(3,4)
        for owner in [1,2]:
            initial = xor_table({},owner,p)
            self.assertTrue(rank_one(initial)[0])
            self.assertEqual(exact_table_risk(initial),F(1,2))
            for value in [0,1]:
                self.assertEqual(exact_table_risk(initial,value),F(1,2))
                later = xor_table({3-owner:value},owner,p)
                self.assertFalse(rank_one(later)[0])
                for c in [0,1]:
                    self.assertEqual(exact_table_risk(later)-exact_table_risk(later,c),p-F(1,2))
        # EVERY pair is independent even after Y; permanent pairwise-MI zero
        # elimination would remove both players and lose v({1,2})=1/4.
        for a,b in itertools.combinations(range(3),2):
            table = [[{F(0):F(0)} for _ in range(2)] for _ in range(2)]
            for w in itertools.product(range(2),repeat=3):
                table[w[a]][w[b]][F(0)] += (p if w[0]^w[1]^w[2] else 1-p)/8
            self.assertTrue(rank_one(table)[0])

    def test_incremental_path_recomputes_after_a_certified_zero_then_positive(self):
        # Encode the RR-XOR posterior as a finite mixture ONLY to exercise the
        # supported COUNT/SUM tree path. Original iid-prior/RR mechanism itself
        # is checked above; this fixture does not extend supported mechanisms.
        worlds=list(itertools.product(range(2),repeat=3))
        mixture=np.array([3/16 if w[0]^w[1]^w[2] else 1/16 for w in worlds])
        probs=[np.array([[int(w[i]==0),int(w[i]==1)] for w in worlds]) for i in range(3)]
        model=Model(probs,mixture,[np.zeros((2,1),dtype=int)]*3)
        release=Release((0.,),(1.,))
        skip,plain=Compiled(model,release),Compiled(model,release)
        for world in worlds:
            for order in [(1,2),(2,1)]:
                observed,certificates=path(skip,world,order,safe_prune=True)
                expected,_=path(plain,world,order)
                np.testing.assert_allclose(observed,expected,atol=1e-12,rtol=0)
                self.assertEqual([c['owner'] for c in certificates],[order[0]])
                self.assertAlmostEqual(observed[order[0]],0.)
                self.assertAlmostEqual(observed[order[1]],.25)
                self.assertEqual(skip.evidence,{i:world[i] for i in order})

    def test_prior_only_separation_is_unsound_for_conditioned_count(self):
        model = Model([np.array([[.5,.5]])]*2,np.array([1.]),[np.array([[0],[1]])]*2)
        prior = [[{F(0):F(1,4)} for _ in range(2)] for _ in range(2)]
        self.assertTrue(rank_one(prior)[0])
        table,_ = PrefixSeparation(model,Release((1,),(1,),kind='exact')).joint_table(1,{})
        self.assertFalse(rank_one(table)[0])
        self.assertEqual(exact_table_risk(table),F(1,2))
        self.assertEqual(exact_table_risk(table,0),F(0))
        # A genuine epsilon=1 Laplace DP release induces dependence as well.
        noisy = Release((1.,),(1.,))
        noisy_table,_ = PrefixSeparation(model,noisy).joint_table(1,{})
        self.assertFalse(rank_one(noisy_table)[0])
        reference = Enumerated(model,noisy)
        self.assertAlmostEqual(reference.risk(),.5)
        reference.update(1,0)
        self.assertAlmostEqual(reference.risk(),1/(1+np.exp(1.)))

    def test_smaller_query_statistic_requires_actual_loss_sufficiency(self):
        model = statistic_counterexample();release = Release((.25,),(1.,))
        a_statistic = (0,0,1,1); b_statistic = model.target_labels
        zero_one_b = [[int(action!=label) for label in b_statistic] for action in [0,1]]
        self.assertFalse(loss_sufficient(a_statistic,zero_one_b))
        self.assertTrue(loss_sufficient(b_statistic,zero_one_b))
        with self.assertRaises(ValueError):PrefixSeparation(model,release,statistic=a_statistic)
        with self.assertRaises(ValueError):PrefixSeparation(model,release,statistic=a_statistic,loss_rows=zero_one_b)
        full = PrefixSeparation(model,release)
        self.assertFalse(full.certificate(1,{})['safe_zero_current_state'])
        self.assertFalse(PrefixSeparation(model,release,statistic=b_statistic,loss_rows=zero_one_b).certificate(1,{})['safe_zero_current_state'])
        table,_ = full.joint_table(1,{})
        a_table = [[{} for _ in range(2)] for _ in range(2)]
        for x,row in enumerate(table):
            for c,cell in enumerate(row):
                for r,v in cell.items():
                    a_table[a_statistic[x]][c][r] = a_table[a_statistic[x]][c].get(r,F(0))+v
        self.assertTrue(rank_one(a_table)[0])
        reference = Enumerated(model,release)
        self.assertAlmostEqual(reference.risk(),.5)
        reference.update(1,0);self.assertAlmostEqual(reference.risk(),0.)

    def test_exact_support_holes_tiny_minors_and_zero_support(self):
        self.assertFalse(rank_one([[{F(0):F(1)},{}],[{},{F(0):F(1)}]])[0])
        self.assertFalse(rank_one([[{F(0):F(1)},{F(0):F(1)}],
                                  [{F(0):F(1)},{F(0):F(1)+F(1,2**100)}]])[0])
        with self.assertRaises(ImpossibleObservation):rank_one([[{},{}],[{},{}]])
        self.assertTrue(rank_one([[{F(0):F(1)},{}],[{F(0):F(2)},{}]])[0])

    def test_deferred_updates_equal_no_pruning_every_path_and_later_reads(self):
        models = [latent_count_model(4,2,531,strength=0),latent_count_model(4,2,531)]
        for model in models:
            for release in [Release((1.2,.7),(.5,.5)),Release((-1.,-1.),(1.,1.))]:
                skip,plain = Compiled(model,release),Compiled(model,release)
                for world in itertools.product(range(2),repeat=4):
                    for order in itertools.permutations(range(1,4)):
                        for loss in ['zero_one','brier']:
                            observed,_ = path(skip,world,order,loss,safe_prune=True)
                            expected,_ = path(plain,world,order,loss)
                            np.testing.assert_allclose(observed,expected,atol=1e-12,rtol=0)
                            np.testing.assert_allclose(skip.posterior_target(),plain.posterior_target(),atol=1e-12,rtol=0)
                            self.assertEqual(skip.evidence,plain.evidence)

    def test_stale_certificate_invalid_values_budget_and_release_retarget(self):
        model = latent_count_model(4,2,531,strength=0)
        release = Release((-1.,-1.),(1.,1.))
        engine = Compiled(model,release)
        cert = engine.pruning_certificate(1)
        self.assertTrue(cert['safe_zero_current_state'])
        engine.update_certified_zero(1,0,cert)
        with self.assertRaises(ValueError):engine.update_certified_zero(2,0,cert)
        # Fresh certificate sees the logically disclosed value even before flush.
        self.assertEqual(engine.pruning_certificate(2)['evidence'],{1:0})
        engine.change_release(Release((1.2,.7),(.5,.5)))
        self.assertEqual(engine.pruning_certificate(2)['release'],engine.release.identity)
        engine.retarget(3)
        self.assertEqual(engine.pruning_certificate(2)['target'],3)
        with self.assertRaises(CertificationBudgetExceeded):rank_one(
            [[{F(0):F(1)},{F(0):F(1)}],[{F(0):F(1)},{F(0):F(1)}]],max_products=0)
        refuser = PrefixSeparation(model,release);refuser.max_pairs=0
        self.assertFalse(refuser.certificate(1,{})['safe_zero_current_state'])
        self.assertEqual(refuser.counters['refused'],1)
        # A failed update can already have set logical evidence. Future calls
        # must never certify using the companion's partially updated messages.
        broken = PrefixSeparation(model,release)
        with patch.object(broken.tree,'_evaluate',side_effect=CertificationBudgetExceeded('forced node budget')):
            self.assertFalse(broken.certificate(2,{1:0})['safe_zero_current_state'])
        self.assertFalse(broken.certificate(2,{1:0})['safe_zero_current_state'])
        self.assertEqual(broken.counters['refused'],2)
        deterministic = Model([np.array([[.5,.5]]),np.array([[1.,0]])],np.array([1.]),
                              [np.array([[0],[1]]),np.array([[0],[0]])])
        safe = Compiled(deterministic,Release((0.,),(1.,)))
        certificate = safe.pruning_certificate(1)
        with self.assertRaises(ImpossibleObservation):safe.update_certified_zero(1,1,certificate)


if __name__ == '__main__':
    unittest.main()
