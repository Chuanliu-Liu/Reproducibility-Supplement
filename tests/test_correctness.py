import itertools
import math
import unittest
import numpy as np
from disclosure.oracle import normalize_worlds, exact_game, realized_risk, ImpossibleObservation
from disclosure.model import Model, Release, latent_count_model
from disclosure.compiled import Compiled, StateBudgetExceeded
from disclosure.baselines import Enumerated
from disclosure.estimate import path, draws, estimate, hoeffding_radius
from disclosure.factors import FactorWorlds,compile_or_fallback


class Correctness(unittest.TestCase):
    def test_general_dependent_prior_xor_separation_and_fallback(self):
        for eps in [0.,1.,4.]:
            p=1/(1+math.exp(-eps))
            lk=np.array([p if (a^b^c)==1 else 1-p for a,b,c in itertools.product(range(2),repeat=3)]).reshape(2,2,2)
            eng=FactorWorlds([2]*3,[((i,),[.5,.5]) for i in range(3)]+[((0,1,2),lk)])
            self.assertFalse(eng.separated(1))
            eng.update(2,0);self.assertFalse(eng.separated(1))
            self.assertAlmostEqual(eng.risk(),.5)
            eng.update(1,0);self.assertAlmostEqual(eng.risk(),1-p)
        # A genuine pairwise dependent prior + a high-order observed release.
        prior=np.array([[.4,.1],[.1,.4]])
        eng=FactorWorlds([2]*3,[((0,1),prior),((2,),[.5,.5]),((0,1,2),lk)])
        post,_=normalize_worlds([2]*3,lambda w:math.log(prior[w[0],w[1]]*.5),lambda w:math.log(lk[w]))
        for a,b in itertools.product(range(2),repeat=2):
            eng.reset();eng.update(1,a);eng.update(2,b)
            self.assertAlmostEqual(eng.risk(),realized_risk(post,0,{1:a,2:b}))
        m=latent_count_model(4,2);r=Release((2,1),(1,1))
        fallback,meta=compile_or_fallback(m,r,max_node_cells=5)
        self.assertEqual(meta['method'],'general_exact_enumeration_fallback')
        self.assertAlmostEqual(fallback.risk(),Compiled(m,r).risk())
    def test_rr_xor_all_coalitions_eps_and_observations(self):
        for eps in [0., .1, 1., 4., 20., 1000.]:
            p = 1/(1+math.exp(-eps))
            for y in [0,1]:
                def likelihood(w):
                    a = p if (w[0]^w[1]^w[2]) == y else 1-p
                    return math.log(a) if a else -math.inf
                post, _ = normalize_worlds([2]*3, lambda w:-3*math.log(2),likelihood)
                self.assertAlmostEqual(sum(x for _,x in post),1)
                game = exact_game(post,0)
                self.assertAlmostEqual(game['risks'][()],.5)
                self.assertAlmostEqual(game['risks'][(1,)],.5)
                self.assertAlmostEqual(game['risks'][(2,)],.5)
                self.assertAlmostEqual(game['risks'][(1,2)],1-p)
                self.assertAlmostEqual(game['phi'][1],(p-.5)/2)
                self.assertAlmostEqual(game['phi'][2],(p-.5)/2)
                self.assertAlmostEqual(game['efficiency_residual'],0)
                # Unsafe pairwise/current unconditional MI pruning removes both.
                self.assertGreaterEqual(game['phi'][1],0)
                if eps > 0:
                    self.assertGreater(game['phi'][1],1e-10)

    def test_exhaustive_compiler_vs_independent_oracle_all_disclosures(self):
        for n,q,seed in [(3,1,4),(4,2,9),(5,2,1),(6,1,15),(8,1,2),(10,1,3)]:
            m=latent_count_model(n,q,seed)
            rel=Release(tuple(n*.3 for _ in range(q)),tuple(1.3 for _ in range(q)))
            posterior,_=normalize_worlds(m.domains,m.log_prior,lambda w:float(rel.log_likelihood(m.counts(w))))
            comp=Compiled(m,rel); ref=Enumerated(m,rel)
            # All worlds/coalitions/disclosure values up to n=6; all coalitions
            # and their probabilities still exhaustive for independent oracle n=10.
            if n <= 6:
                for states in itertools.product(*[[-1,*range(d)] for d in m.domains[1:]]):
                    ev={i+1:v for i,v in enumerate(states) if v>=0}
                    comp.reset(); ref.reset()
                    for i,v in ev.items():comp.update(i,v);ref.update(i,v)
                    for loss in ['zero_one','brier']:
                        risk=realized_risk(posterior,0,ev,loss=loss)
                        self.assertAlmostEqual(comp.risk(loss),risk,places=11)
                        self.assertAlmostEqual(ref.risk(loss),risk,places=11)
            game=exact_game(posterior,0)
            self.assertAlmostEqual(game['efficiency_residual'],0,places=11)
            for s,r in game['risks'].items():
                for i in range(1,n):
                    if i not in s:self.assertLessEqual(game['risks'][tuple(sorted((*s,i)))],r+1e-11)

    def test_same_draws_cache_naive_and_reset(self):
        m=latent_count_model(9,2,33)
        rel=Release((3.2,2.8),(1.,1.8),derived_answers=((0,2.,0.,6.4),))
        cached=Compiled(m,rel);naive=Compiled(m,rel,cache=False);ref=Enumerated(m,rel)
        d=draws(cached,32,77)
        for loss in ['zero_one','brier']:
            for w,o in d:
                a,_=path(cached,w,o,loss);b,_=path(naive,w,o,loss);c,_=path(ref,w,o,loss)
                np.testing.assert_allclose(a,b,atol=1e-11)
                np.testing.assert_allclose(a,c,atol=1e-11)
        cached.reset();self.assertEqual(cached.evidence,{})
        np.testing.assert_allclose(cached.posterior_target(),Compiled(m,rel).posterior_target(),atol=1e-12)
        new=Release((1.1,5.9),(1.,1.8),release_id='changed')
        cached.change_release(new)
        np.testing.assert_allclose(cached.posterior_target(),Compiled(m,new).posterior_target(),atol=1e-12)
        self.assertNotEqual(cached.key[1],rel.identity)

    def test_multicategory_correlated_bundle_signed_sum(self):
        probs=[np.array([[.2,.3,.1,.4],[.4,.1,.4,.1]]) for _ in range(3)]
        # One bundle contains two correlated binary variables; player is unsplit.
        c=[np.array([[0,-1],[1,0],[1,1],[2,2]]) for _ in probs]
        m=Model(probs,np.array([.6,.4]),c,target_labels=(0,0,1,1))
        r=Release((2.1,-.5),(2.,3.))
        p,_=normalize_worlds(m.domains,m.log_prior,lambda w:float(r.log_likelihood(m.counts(w))))
        eng=Compiled(m,r)
        for a,b in itertools.product(range(4),repeat=2):
            eng.reset();eng.update(1,a);eng.update(2,b)
            self.assertAlmostEqual(eng.risk('brier'),realized_risk(p,0,{1:a,2:b},m.target_labels,'brier'),places=11)

    def test_negative_increment_and_prior_sampling_failure(self):
        m=latent_count_model(5,1,18,strength=.32)
        r=Release((.15,),(.25,))
        p,_=normalize_worlds(m.domains,m.log_prior,lambda w:float(r.log_likelihood(m.counts(w))))
        before=realized_risk(p,0,{})
        diffs=[before-realized_risk(p,0,{1:v}) for v in [0,1]]
        self.assertTrue(any(d < -1e-5 for d in diffs))
        posterior_expected=sum(pr*(before-realized_risk(p,0,{1:w[1]})) for w,pr in p)
        prior_expected=sum(math.exp(m.log_prior(w))*(before-realized_risk(p,0,{1:w[1]}))
                           for w in itertools.product(*[range(d) for d in m.domains]))
        self.assertGreater(abs(prior_expected-posterior_expected),1e-4)
        clipped=sum(pr*max(0,before-realized_risk(p,0,{1:w[1]})) for w,pr in p)
        self.assertGreater(clipped-posterior_expected,1e-5)

    def test_exact_backward_sampler_full_world_distribution(self):
        m=latent_count_model(4,2,5)
        r=Release((1.2,.5),(.9,1.1))
        ref=Enumerated(m,r);eng=Compiled(m,r);rng=np.random.default_rng(442)
        counts={tuple(w):0 for w in ref.worlds}
        for _ in range(12000):counts[tuple(eng.sample_world(rng))]+=1
        empirical=np.array([counts[tuple(w)]/12000 for w in ref.worlds])
        self.assertLess(np.abs(empirical-ref.weights).sum()/2,.035)
        # Posterior differs substantially from prior in this regression.
        prior=np.array([math.exp(m.log_prior(w)) for w in ref.worlds])
        self.assertGreater(np.abs(empirical-prior).sum()/2,.1)

    def test_safe_pruning_full_graph_and_future_synergy(self):
        m=latent_count_model(4,1,5,strength=0)
        m.contributions[3]=np.array([[0],[0]])
        r=Release((1.4,),(1.,))
        eng=Compiled(m,r)
        self.assertFalse(eng.pruning_certificate(1)['safe_zero_current_state'])
        self.assertTrue(eng.pruning_certificate(3)['safe_zero_current_state'])
        for w,o in draws(eng,16,88):
            a,_=path(eng,w,o);b,cert=path(eng,w,o,safe_prune=True)
            np.testing.assert_allclose(a,b,atol=1e-12)
            self.assertTrue(cert)
        # Observed count connects owners even if independent in prior.
        self.assertGreater(abs(Compiled(m,r).risk()-Compiled(m,Release((0,),(1e15,))).risk()),1e-3)

    def test_zero_support_underflow_and_budget(self):
        with self.assertRaises(ImpossibleObservation):normalize_worlds([2],lambda w:0,lambda w:-math.inf)
        m=latent_count_model(3)
        with self.assertRaises(ImpossibleObservation):Compiled(m,Release((100.,),(1.,),kind='exact')).risk()
        with self.assertRaises(ValueError):Release((2.,),(1.,),derived_answers=((0,2,0,5),))
        # Extreme density values remain finite after logarithmic rescaling.
        self.assertTrue(np.isfinite(Compiled(m,Release((10000.,),(.1,))).risk()))
        with self.assertRaises(StateBudgetExceeded):Compiled(latent_count_model(64,2),Release((10,10),(1,1)),max_node_cells=100)
        with self.assertRaises(MemoryError):Enumerated(latent_count_model(24),Release((4,),(1,)))
        self.assertEqual(hoeffding_radius(1,0),0)

    def test_rare_fft_tail_repair_and_changed_release_support(self):
        m=latent_count_model(128,1,13)
        e=Compiled(m,Release((0.,),(1.,),kind='exact'))
        np.testing.assert_allclose(e.posterior_target(),[1.,0.],atol=1e-13)
        rng=np.random.default_rng(133)
        for _ in range(8):self.assertTrue(np.all(e.sample_world(rng)==0))
        e.change_release(Release((128.,),(1.,),kind='exact'))
        np.testing.assert_allclose(e.posterior_target(),[0.,1.],atol=1e-13)
        for _ in range(8):self.assertTrue(np.all(e.sample_world(rng)==1))
        # Rare INTERIOR exact count, independently evaluated singleton worlds.
        e.change_release(Release((1.,),(1.,),kind='exact'))
        logs=[];target_logs=[]
        for owner in range(128):
            world=np.zeros(128,dtype=int);world[owner]=1
            v=m.log_prior(world);logs.append(v)
            if owner==0:target_logs.append(v)
        from scipy.special import logsumexp
        expected=np.exp(logsumexp(target_logs)-logsumexp(logs))
        self.assertAlmostEqual(e.posterior_target()[1],expected,places=10)
        for _ in range(8):self.assertEqual(e.sample_world(rng).sum(),1)
        for y in [-1000.,1000.]:
            small=latent_count_model(8,2,13)
            r=Release((y,y),(.25,.4))
            p,_=normalize_worlds(small.domains,small.log_prior,lambda w:float(r.log_likelihood(small.counts(w))))
            e=Compiled(small,r)
            self.assertAlmostEqual(e.risk(),realized_risk(p,0,{}),places=10)
            for i in range(1,8):
                e.update(i,1 if y>0 else 0)
                self.assertAlmostEqual(e.risk(),realized_risk(p,0,{j:(1 if y>0 else 0) for j in range(1,i+1)}),places=10)

    def test_retarget_reuses_unaffected_scopes_and_identity(self):
        from dataclasses import replace
        model=latent_count_model(8,2,63);rel=Release((3.4,1.3),(1.,1.5))
        e=Compiled(model,rel);e.update(7,1)
        for target in [3,1,6,0]:
            prior_evals=e.counters['evaluations']
            e.retarget(target)
            fresh=Compiled(replace(model,target=target),rel);fresh.update(7,1)
            np.testing.assert_allclose(e.posterior_target(),fresh.posterior_target(),atol=1e-12)
            self.assertLess(e.counters['evaluations']-prior_evals,len(e.nodes))
            self.assertEqual(e.key[2],target)
        with self.assertRaises(ValueError):e.retarget(7)

    def test_latent_mixture_interior_tilt_against_positive_direct_count(self):
        model=latent_count_model(128,1,881);release=Release((128*.4,),(1.,))
        joint=np.zeros(2)
        for h,weight in enumerate(model.mixture):
            poly=np.array([1.])
            for i,p in enumerate(model.probs):
                if i!=0:poly=np.convolve(poly,p[h])
            for target in [0,1]:
                likelihood=np.exp(release.log_likelihood((np.arange(len(poly))+target)[:,None]))
                joint[target]+=weight*model.probs[0][h,target]*float(poly@likelihood)
        e=Compiled(model,release)
        np.testing.assert_allclose(e.posterior_target(),joint/joint.sum(),atol=1e-10)
        rng=np.random.default_rng(197)
        for _ in range(16):self.assertEqual(len(e.sample_world(rng)),128)

    def test_integer_contrast_basis_preserves_full_release_and_sampling(self):
        model=latent_count_model(6,2,531)
        model.contributions=[np.array([[0,0],[4,4]]) for _ in range(6)]
        model.contributions[5]=np.array([[0,0],[3,4]])
        release=Release((11.2,11.8),(2.,3.))
        engine=Compiled(model,release);reference=Enumerated(model,release)
        self.assertLess(engine.basis_diagnostic['compiled_sum_box_cells'],engine.basis_diagnostic['original_sum_box_cells'])
        rng=np.random.default_rng(849)
        for _ in range(20):
            engine.reset()
            w=engine.sample_world(rng);order=rng.permutation(np.arange(1,6))
            a,_=path(engine,w,order);b,_=path(reference,w,order)
            np.testing.assert_allclose(a,b,atol=1e-11)


if __name__=='__main__':unittest.main(verbosity=2)
