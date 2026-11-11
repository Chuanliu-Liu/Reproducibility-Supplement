import itertools
import unittest
from fractions import Fraction as F
from decimal import Decimal
import numpy as np
from scipy.special import logsumexp
from disclosure.absorption import AbsorptionCompiled, support_sign
from disclosure.baselines import Enumerated
from disclosure.compiled import Compiled
from disclosure.exact_tree import exp_minus_bounds
from disclosure.factors import FactorWorlds, compile_or_fallback
from disclosure.model import Model, Release, latent_count_model
from disclosure.oracle import ImpossibleObservation
from disclosure.estimate import path
from test_r3_certificates import signed_model, MATRICES


def symbolic_worlds(model, release, evidence, slopes=None):
    """Independent full-world expansion; rational coefficients/exponents.

    Not product-tree enumeration. Absorbed expansion retains unnormalized u
    per class, so both the coefficient and exact exponent are compared.
    """
    out = {}
    slopes = {} if slopes is None else slopes
    for w in itertools.product(*(range(d) for d in model.domains)):
        if any(w[i] != v for i,v in evidence.items()):continue
        counts = [sum(int(c[v,d]) for c,v in zip(model.contributions,w))
                  for d in range(len(release.observed))]
        exponent = F(0)
        for d,(y,b) in enumerate(zip(release.observed,release.scales)):
            y,b = F(float(y)),F(float(b))
            if d in slopes:
                slope=slopes[d]
                exponent += sum(slope*int(c[v,d]) for c,v in zip(model.contributions,w))
                exponent += -y/b if slope>0 else y/b
            else:exponent -= abs(y-counts[d])/b
        for h,pi in enumerate(model.mixture):
            coefficient=F(float(pi))
            for p,v in zip(model.probs,w):coefficient *= F(float(p[h,v]))
            if coefficient:out[(w,h)] = (coefficient,exponent)
    if not out:raise ImpossibleObservation('Independent zero support')
    return out


def entire_posterior_enclosure(expansion,bits=100):
    shift=max(exponent for coefficient,exponent in expansion.values())
    masses={}
    for (w,h),(c,e) in expansion.items():
        lo,hi=exp_minus_bounds(shift-e,bits)
        old=masses.get(w,(F(0),F(0)));masses[w]=(old[0]+c*lo,old[1]+c*hi)
    lower=sum(m[0] for m in masses.values());upper=sum(m[1] for m in masses.values())
    return {w:(lo/upper,hi/lower) for w,(lo,hi) in masses.items()}


class AbsorptionTests(unittest.TestCase):
    def test_complete_symbolic_world_measures_and_enclosed_target(self):
        self.checked_world_class_masses=0
        for model in [signed_model(),latent_count_model(4,2,41),latent_count_model(3,1,41)]:
            q=model.contributions[0].shape[1]
            lower=np.sum([c.min(axis=0) for c in model.contributions],axis=0)
            upper=np.sum([c.max(axis=0) for c in model.contributions],axis=0)
            ys=[tuple(lower-1),tuple(upper+1),tuple(lower),tuple(upper),tuple((lower+upper)/2)]
            players=[i for i in range(len(model.probs)) if i!=model.target]
            for y in ys:
                r=Release(tuple(map(float,y)),tuple([1.25]*q))
                reference=Enumerated(model,r)
                for states in itertools.product(*[[-1,*range(model.domains[i])] for i in players]):
                    e=AbsorptionCompiled(model,r,convolution_method='direct');reference.reset()
                    for i,v in zip(players,states):
                        if v>=0:e.update(i,v);reference.update(i,v)
                    original=symbolic_worlds(model,r,e.evidence)
                    compiled=symbolic_worlds(model,r,e.evidence,e.slopes)
                    self.assertEqual(original,compiled)
                    self.checked_world_class_masses += len(original)
                    enclosures=entire_posterior_enclosure(original)
                    # Reconstruct the ENTIRE posterior represented by the actual
                    # compiled factors/class weights, independently of its tree.
                    worlds=list(enclosures)
                    represented=[]
                    for w in worlds:
                        if e.backend is not None:
                            represented.append(e.backend.model.log_prior(w)+float(
                                e.backend.release.log_likelihood(e.backend.model.counts(w))))
                        else:
                            logs=e.base_logweights.copy()
                            for i,v in enumerate(w):logs+=e.log_probabilities[i][:,v]
                            represented.append(float(logsumexp(logs)))
                    represented=np.exp(np.array(represented)-logsumexp(represented))
                    for w,pw in zip(worlds,represented):
                        lo,hi=enclosures[w]
                        self.assertGreaterEqual(pw,float(lo)-3e-12)
                        self.assertLessEqual(pw,float(hi)+3e-12)
                    expected=np.zeros(max(model.target_labels)+1)
                    lower_p=np.zeros_like(expected);upper_p=np.zeros_like(expected)
                    for w,(lo,hi) in enclosures.items():
                        a=model.target_labels[w[model.target]]
                        lower_p[a]+=float(lo);upper_p[a]+=float(hi)
                        expected[a]+=(float(lo)+float(hi))/2
                    p=e.posterior_target()
                    self.assertTrue(np.all(p>=lower_p-3e-12) and np.all(p<=upper_p+3e-12))
                    np.testing.assert_allclose(p,reference.posterior_target(),atol=3e-12,rtol=0)

    def test_two_one_zero_phases_exact_boundary_and_signed_increments(self):
        model=Model([np.array([[.5,.5]])]*3,np.array([1.]),
            [np.array([[0,0],[1,0]]),np.array([[0,0],[1,0]]),np.array([[0,0],[0,2]])])
        r=Release((1.,1.),(1.,1.));e=AbsorptionCompiled(model,r)
        self.assertEqual(e.active_dimension,2)
        e.update(1,0);self.assertEqual(e.active_dimension,1)
        e.update(2,0);self.assertEqual(e.active_dimension,0)
        self.assertEqual([p['active_coordinates'] for p in e.phase_history],[[0,1],[1],[]])
        self.assertEqual(e.counters['eliminated_coordinates'],2)
        for world in itertools.product([0,1],repeat=3):
            for order in itertools.permutations([1,2]):
                for loss in ['zero_one','brier']:
                    a,_=path(e,np.array(world),order,loss)
                    b,_=path(Compiled(model,r,convolution_method='direct'),np.array(world),order,loss)
                    np.testing.assert_allclose(a,b,atol=2e-12,rtol=0)
        interior=np.nextafter(1.,0.)
        self.assertEqual(support_sign(interior,0,1),0)
        self.assertEqual(support_sign(1.,0,1),1)
        self.assertEqual(support_sign(np.nextafter(0.,1.),0,1),0)
        self.assertEqual(support_sign(0.,0,1),-1)
        self.assertEqual(support_sign(float('nan'),0,1),0)
        self.assertEqual(support_sign(Decimal('0.999999999999999999999'),0,1),0)
        self.assertEqual(support_sign(F(1)-F(1,10**30),0,1),0)
        self.assertEqual(support_sign('0.999999999999999999999',0,1),0)
        self.assertEqual(support_sign(np.longdouble('0.9999999999999999999'),0,1),0)

    def test_class_normalizers_negative_control_and_remaining_coupling(self):
        p=[np.array([[.9,.1],[.1,.9]]),np.array([[.8,.2],[.2,.8]])]
        model=Model(p,np.array([.5,.5]),[np.array([[0],[1]]),np.array([[0],[2]])])
        e=AbsorptionCompiled(model,Release((-1.,),(1.,)))
        ref=Enumerated(model,e.release)
        np.testing.assert_allclose(e.posterior_target(),ref.posterior_target(),atol=1e-12)
        wrong=model.mixture@e.probabilities[0]  # Intentionally DROP all A_ih.
        self.assertGreater(np.max(abs(wrong-e.posterior_target())),.05)
        model=Model(p,np.array([.5,.5]),[np.array([[0],[0]])]*2)
        e=AbsorptionCompiled(model,Release((-1.,),(1.,)))
        self.assertEqual(e.active_dimension,0)
        before=e.risk();e.update(1,1)
        self.assertGreater(before-e.risk(),.1)

    def test_zero_classes_support_impossible_nonbinary_and_sufficient_labels(self):
        model=Model([np.array([[.25,.25,.5],[.5,.5,0],[0,0,1]]),
                     np.array([[1,0],[0,1],[1,0]])],np.array([.5,.5,0]),
                    [np.array([[-1],[0],[2]]),np.array([[0],[1]])],target_labels=(0,0,1))
        e=AbsorptionCompiled(model,Release((-3.,),(.75,)))
        ref=Enumerated(model,e.release)
        for v in [0,1]:
            e.reset();ref.reset();e.update(1,v);ref.update(1,v)
            np.testing.assert_allclose(e.posterior_target(),ref.posterior_target(),atol=1e-12)
        impossible=Model([np.array([[.5,.5]])]*2+[np.array([[1,0]])],np.array([1.]),
                         [np.array([[0],[1]])]*3)
        e=AbsorptionCompiled(impossible,Release((-1.,),(1.,)))
        with self.assertRaises(ImpossibleObservation):e.update(2,1)
        self.assertEqual(e.evidence,{})

    def test_reset_release_retarget_and_original_unimodular_coordinates(self):
        model=signed_model();release=Release((3.,.5),(.75,1.25))
        for matrix in MATRICES:
            e=AbsorptionCompiled(model,release,query_matrix=matrix)
            ref=Enumerated(model,release)
            for i,v in [(1,1),(2,0)]:e.update(i,v);ref.update(i,v)
            np.testing.assert_allclose(e.posterior_target(),ref.posterior_target(),atol=1e-12)
            with self.assertRaises(ValueError):e.update(1,0)
            e.reset();e.update(1,1)
            r2=Release((-4.,-4.),(1.,1.));e.change_release(r2)
            ref=Enumerated(model,r2);ref.update(1,1)
            np.testing.assert_allclose(e.posterior_target(),ref.posterior_target(),atol=1e-12)
            e.retarget(2);reference=Enumerated(e.model,r2);reference.update(1,1)
            np.testing.assert_allclose(e.posterior_target(),reference.posterior_target(),atol=1e-12)
            self.assertGreater(e.generation,1)
            with self.assertRaises(ValueError):e.retarget(1)

    def test_exact_delta_and_rr_xor_decline_shared_synopsis(self):
        model=latent_count_model(3,2,41)
        r=Release((1.,1.),(1.,1.),kind='exact');e=AbsorptionCompiled(model,r)
        self.assertEqual(e.slopes,{})
        np.testing.assert_allclose(e.posterior_target(),Enumerated(model,r).posterior_target(),atol=1e-12)
        with self.assertRaises(ValueError):
            AbsorptionCompiled(model,Release((1.,1.),(1.,1.),kind='correlated_laplace'))
        with self.assertRaises(ValueError):Release((1.,),(1.,),derived_answers=((0,2,0,3),))
        r1=Release((-1.,-1.),(1.,1.),derived_answers=((0,2,0,-2),))
        r2=Release((-1.,-1.),(1.,1.))
        np.testing.assert_array_equal(AbsorptionCompiled(model,r1).posterior_target(),AbsorptionCompiled(model,r2).posterior_target())
        xor=np.empty((2,2,2))
        for w in itertools.product([0,1],repeat=3):xor[w]=.75 if w[0]^w[1]^w[2] else .25
        fallback=FactorWorlds([2]*3,[(tuple(range(3)),xor)])
        before=fallback.risk();fallback.update(1,0);self.assertAlmostEqual(before-fallback.risk(),0.)
        fallback.update(2,0);self.assertAlmostEqual(before-fallback.risk(),.25)
        engine,metadata=compile_or_fallback(model,Release((-1.,-1.),(1.,1.)))
        self.assertIsInstance(engine,AbsorptionCompiled)

    def test_complete_world_sampling_q0_and_q1_and_negative_increments(self):
        model=latent_count_model(4,2,41)
        for y in [(-1.,-1.),(-1.,1.)]:
            e=AbsorptionCompiled(model,Release(y,(1.,1.)))
            reference=Enumerated(model,e.release);counts=np.zeros(len(reference.worlds))
            lookup={tuple(w):k for k,w in enumerate(reference.worlds)}
            rng=np.random.default_rng(612)
            for _ in range(4000):counts[lookup[tuple(e.sample_world(rng))]]+=1
            self.assertLess(max(abs(counts/4000-reference.weights)),.04)
        e=AbsorptionCompiled(model,Release((1.5,.75),(1.,1.)))
        negative=0
        for w in itertools.product([0,1],repeat=4):
            for order in itertools.permutations([1,2,3]):
                inc,_=path(e,np.array(w),order)
                expected,_=path(Enumerated(model,e.release),np.array(w),order)
                np.testing.assert_allclose(inc,expected,atol=1e-12)
                negative+=int((inc<-1e-12).sum())
        self.assertGreater(negative,0)


if __name__ == '__main__':unittest.main()
