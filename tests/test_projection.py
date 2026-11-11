"""Independent rational/exponential C6 algebra and finite production checks."""
import itertools
import unittest
from fractions import Fraction as F
import numpy as np
from disclosure.projection import ProjectionCompiled
from disclosure.compiled import Compiled
from disclosure.baselines import Enumerated
from disclosure.model import Model, Release, latent_count_model
from disclosure.coordinates import UnimodularCertificate, transform
from disclosure.absorption import support_sign
from disclosure.exact_tree import exp_minus_bounds
from disclosure.estimate import path
from disclosure.factors import FactorWorlds
from test_absorption import symbolic_worlds, entire_posterior_enclosure

COUNTS = dict(homomorphism_entries=0, fresh_message_equalities=0,
    world_class_equalities=0, posterior_cells=0, production_prefixes=0,
    signed_negative_increments=0, max_float_error=0., max_mass_error=0.)
COUNTS.update(physical_root_entry_enclosures=0,max_scaled_root_entry_error=0.)


def add(table,key,exponent,coefficient):
    if coefficient:
        cell = table.setdefault(key,{})
        cell[exponent] = cell.get(exponent,F(0))+coefficient


def convolution(left,right,left_target=False):
    out = {}
    for (h,a,s),x in left.items():
        for (g,b,t),y in right.items():
            if h != g:
                continue
            key = (h,a if left_target else b,tuple(u+v for u,v in zip(s,t)))
            for e,c in x.items():
                for f,d in y.items():
                    add(out,key,e+f,c*d)
    return out


def projection(table,active,inverse,eliminated,delta):
    out = {}
    retain = [k for k,d in enumerate(active) if d not in eliminated]
    for (h,a,z),cell in table.items():
        s = transform(inverse,z)
        key = (h,a,tuple(s[k] for k in retain))
        correction = sum(delta[h][k]*s[k] for k in range(len(active)))
        for exponent,coefficient in cell.items():
            add(out,key,exponent+correction,coefficient)
    return out


def scope_table(model,scope,evidence,active,theta,matrix):
    out = {}
    for world in itertools.product(*(range(model.domains[i]) for i in scope)):
        if any(i in evidence and evidence[i] != v for i,v in zip(scope,world)):
            continue
        sums = [sum(int(model.contributions[i][v,d]) for i,v in zip(scope,world))
                for d in range(model.contributions[0].shape[1])]
        state = world[scope.index(model.target)] if model.target in scope else 0
        label = model.target_labels[state] if model.target in scope else 0
        encoded_matrix = tuple(tuple(int(x) for x in row) for row in matrix)
        z = transform(encoded_matrix,[sums[d] for d in active]) if active else ()
        for h in range(len(model.mixture)):
            coefficient = F(1)
            for i,v in zip(scope,world):
                coefficient *= F(float(model.probs[i][h,v]))
            exponent = sum(theta[h][d]*sums[d] for d in range(len(sums)))
            add(out,(h,label,z),exponent,coefficient)
    return out


def multistate(q=2):
    probs = [np.array([[.25,.25,.5],[.5,.5,0],[0,0,1]]),
             np.array([[.5,.25,.25],[.125,.375,.5],[1,0,0]]),
             np.array([[.5,.5],[0,1],[.25,.75]])]
    c = [np.array([[-1,1],[0,-1],[2,2]]),
         np.array([[0,1],[1,-1],[2,0]]),np.array([[0,0],[1,2]])]
    return Model(probs,np.array([.5,.5,0]),[a[:,:q] for a in c],target_labels=(0,1,2))


MATRICES = [[[1,0],[0,1]],[[1,0],[-1,1]],[[1,-1],[0,1]],[[2,1],[1,1]]]


class ProjectionTests(unittest.TestCase):
    def test_exact_homomorphism_offsets_tilts_multistate_channels(self):
        for q in [1,2]:
            m = multistate(q)
            active = tuple(range(q))
            # Exact rational class-specific tilts, including zero class mass.
            theta = [[F(h+1,7)*(-1 if d else 1) for d in active] for h in range(3)]
            for matrix in ([[1]],) if q == 1 else MATRICES:
                certificate = UnimodularCertificate.create(matrix,q)
                for eliminated in ([0],) if q == 1 else ([0],[1],[0,1]):
                    newtheta = [t.copy() for t in theta]
                    for h in range(3):
                        for d in eliminated:newtheta[h][d] = F(4-d,5)
                    delta = [[newtheta[h][d]-theta[h][d] for d in active] for h in range(3)]
                    retain = tuple(d for d in active if d not in eliminated)
                    for values in itertools.product([-1,0,1,2],[-1,0,1]):
                        evidence = {i:v for i,v in zip([1,2],values) if v >= 0}
                        left = scope_table(m,(0,),evidence,active,theta,certificate.matrix)
                        right = scope_table(m,(1,2),evidence,active,theta,certificate.matrix)
                        lhs = projection(convolution(left,right,True),active,certificate.inverse,eliminated,delta)
                        rhs = convolution(projection(left,active,certificate.inverse,eliminated,delta),
                            projection(right,active,certificate.inverse,eliminated,delta),True)
                        self.assertEqual(lhs,rhs)
                        fresh = scope_table(m,(0,1,2),evidence,retain,newtheta,np.eye(len(retain),dtype=int))
                        self.assertEqual(lhs,fresh)
                        COUNTS['homomorphism_entries'] += len(lhs)
                        COUNTS['fresh_message_equalities'] += 1
                    if q == 2 and len(eliminated) == 2:
                        table = convolution(left,right,True)
                        # First transform must compensate only coordinate 0.
                        d0 = [[row[0],F(0)] for row in delta]
                        first = projection(table,active,certificate.inverse,[0],d0)
                        sequential = projection(first,(1,),((1,),),[1],[[row[1]] for row in delta])
                        self.assertEqual(sequential,projection(table,active,certificate.inverse,[0,1],delta))

    def test_all_valid_prefixes_world_class_masses_and_signed_paths(self):
        for m in [multistate(1),multistate(2),latent_count_model(3,2,41)]:
            q = m.contributions[0].shape[1]
            low = np.sum([c.min(axis=0) for c in m.contributions],axis=0)
            high = np.sum([c.max(axis=0) for c in m.contributions],axis=0)
            ys = [low-1,high+1,low,high,(low+high)/2]
            players = [i for i in range(len(m.probs)) if i != m.target]
            for y in ys:
                release = Release(tuple(map(float,y)),tuple([1.25]*q))
                matrices = [None] if q == 1 else MATRICES
                for matrix in matrices:
                    e = ProjectionCompiled(m,release,query_matrix=matrix,convolution_method='direct')
                    reference = Enumerated(m,release)
                    for order in itertools.permutations(players):
                        for values in itertools.product(*(range(m.domains[i]) for i in players)):
                            evidence = dict(zip(players,values))
                            if not any(all(m.probs[i][h,v] > 0 for i,v in evidence.items()) and m.mixture[h] > 0 for h in range(len(m.mixture))):
                                continue
                            e.reset();reference.reset()
                            for i in [None,*order]:
                                if i is not None:
                                    e.update(i,evidence[i]);reference.update(i,evidence[i])
                                original = symbolic_worlds(m,release,e.evidence)
                                reduced = symbolic_worlds(m,release,e.evidence,e.slopes)
                                self.assertEqual(original,reduced)
                                COUNTS['world_class_equalities'] += len(original)
                                bounds = entire_posterior_enclosure(original,100)
                                lower = np.zeros(m.domains[m.target]);upper = lower.copy()
                                for w,(lo,hi) in bounds.items():
                                    lower[w[m.target]] += float(lo);upper[w[m.target]] += float(hi)
                                p = e.posterior_target()
                                self.assertTrue(np.all(p >= lower-4e-12) and np.all(p <= upper+4e-12))
                                error = float(max(abs(p-reference.posterior_target())))
                                COUNTS['max_float_error'] = max(COUNTS['max_float_error'],error)
                                COUNTS['posterior_cells'] += len(p)
                                COUNTS['production_prefixes'] += 1
                                self.assertLess(error,4e-12)
                                # Complete class/target root measure, not posterior
                                # alone: common likelihood constants retained once.
                                weighted,logz = e._weighted_root()
                                # Tie the independent world/class expansion to
                                # the ACTUAL physical message, including active
                                # sum keys, target channels and stored log mass.
                                theta=[[F(float(v)) for v in row] for row in e.theta]
                                fresh=scope_table(m,tuple(range(len(m.probs))),e.evidence,
                                    e.active,theta,e.query_basis)
                                shifts={h:max(exponent for (g,a,z),cell in fresh.items() if g==h
                                    for exponent in cell) for h in {key[0] for key in fresh}}
                                seen=set()
                                for (h,a,z),cell in fresh.items():
                                    lower_mass=upper_mass=F(0)
                                    for exponent,coefficient in cell.items():
                                        lo,hi=exp_minus_bounds(shifts[h]-exponent,100)
                                        lower_mass+=coefficient*lo;upper_mass+=coefficient*hi
                                    key=(h,a,*tuple(np.array(z)-e.root.offset))
                                    seen.add(key)
                                    actual=float(e.root.poly[key]*np.exp(e.root.logmass[h]-float(shifts[h])))
                                    self.assertGreaterEqual(actual,float(lower_mass)-4e-12)
                                    self.assertLessEqual(actual,float(upper_mass)+4e-12)
                                    COUNTS['physical_root_entry_enclosures']+=1
                                    COUNTS['max_scaled_root_entry_error']=max(COUNTS['max_scaled_root_entry_error'],
                                        abs(actual-(float(lower_mass)+float(upper_mass))/2))
                                for key in np.ndindex(e.root.poly.shape):
                                    if key not in seen and np.isfinite(e.root.logmass[key[0]]):
                                        self.assertLess(abs(float(e.root.poly[key])),4e-12)
                                logs = [m.log_prior(w)+float(release.log_likelihood(m.counts(w)))
                                    for w in itertools.product(*(range(d) for d in m.domains))
                                    if all(w[i] == v for i,v in e.evidence.items())]
                                from scipy.special import logsumexp
                                mass_error = abs(logz-float(logsumexp(logs)))
                                COUNTS['max_mass_error'] = max(COUNTS['max_mass_error'],mass_error)
                                self.assertLess(mass_error,8e-12)
                            world = np.array([0,*values])
                            for loss in ['zero_one','brier']:
                                a,_ = path(e,world,order,loss)
                                b,_ = path(reference,world,order,loss)
                                np.testing.assert_allclose(a,b,atol=4e-12,rtol=0)
                                COUNTS['signed_negative_increments'] += int((a < -1e-12).sum())
        self.assertGreater(COUNTS['signed_negative_increments'],0)

    def test_reset_draw_identity_memory_and_retarget_release(self):
        m = latent_count_model(5,2,41)
        r = Release((1.5,.75),(1.,1.))
        e = ProjectionCompiled(m,r)
        rng1,rng2 = np.random.default_rng(71),np.random.default_rng(71)
        original = Compiled(m,r)
        for _ in range(12):
            np.testing.assert_array_equal(e.sample_world(rng1),original.sample_world(rng2))
            e.update(1,1);e.update(2,0)
            ids = {}
            for n in e.nodes:
                for a in (n.poly,n.offset,n.logmass):ids[id(a)] = a.nbytes
            for arrays in e._snapshot['nodes']:
                for a in arrays:ids[id(a)] = a.nbytes
            cache = sum(a.nbytes+b.nbytes for a,b in e.likelihood_cache.values())
            self.assertEqual(e.memory_bytes,sum(ids.values())+cache+e._original_sampler.memory_bytes)
            e.reset()
            np.testing.assert_allclose(e.posterior_target(),original.posterior_target(),atol=1e-12)
        e.update(1,1);e.retarget(3)
        reference = Enumerated(e.model,r);reference.update(1,1)
        np.testing.assert_allclose(e.posterior_target(),reference.posterior_target(),atol=1e-12)
        e.reset();e.retarget(0)
        e.update(2,0);r2=Release((-3.,-2.),(1.,1.));e.change_release(r2)
        reference=Enumerated(e.model,r2);reference.update(2,0)
        np.testing.assert_allclose(e.posterior_target(),reference.posterior_target(),atol=1e-12)
        e.reset()
        current=Compiled(e.model,r2)
        r1,r2rng=np.random.default_rng(88),np.random.default_rng(88)
        for _ in range(20):
            np.testing.assert_array_equal(e.sample_world(r1),current.sample_world(r2rng))
        e.retarget(4)
        current=Compiled(e.model,e.release)
        r1,r2rng=np.random.default_rng(89),np.random.default_rng(89)
        for _ in range(20):
            np.testing.assert_array_equal(e.sample_world(r1),current.sample_world(r2rng))

    def test_negative_controls_normalizers_tilt_stale_shared_partial_support(self):
        m=multistate(2)
        theta=[[F(1,3),F(-2,5)]]*3
        table=scope_table(m,(0,1,2),{},(0,1),theta,((1,0),(0,1)))
        correct=projection(table,(0,1),((1,0),(0,1)),[0],[[F(2)-F(1,3),F(0)]]*3)
        wrong=projection(table,(0,1),((1,0),(0,1)),[0],[[F(2),F(0)]]*3)
        self.assertNotEqual(correct,wrong)
        fresh=scope_table(m,(0,1,2),{1:1},(1,),[[F(2),F(-2,5)]]*3,((1,),))
        self.assertNotEqual(correct,fresh)  # stale dirty-root projection
        e=ProjectionCompiled(m,Release((-5.,-5.),(1.,1.)))
        wrong_p=m.mixture@e.leaves[0].poly
        self.assertGreater(float(max(abs(wrong_p-e.posterior_target()))),1e-4)
        # A restricted observed support would certify, the declared full support
        # correctly refuses. Correlated/shared likelihood stays on general path.
        self.assertEqual(support_sign(2.,0,1),1)
        self.assertEqual(support_sign(2.,0,3),0)
        with self.assertRaises(ValueError):ProjectionCompiled(m,Release((1.,1.),(1.,1.),kind='correlated_laplace'))
        independent=Release((-1.,-1.),(1.,1.))
        shared=Release((-1.,-1.),(1.,1.),derived_answers=((0,2,0,-2),))
        np.testing.assert_array_equal(ProjectionCompiled(m,independent).posterior_target(),ProjectionCompiled(m,shared).posterior_target())
        # Duplicating the same likelihood changes the model and posterior.
        duplicate=ProjectionCompiled(m,Release((-1.,-1.),(.5,.5)))
        base=ProjectionCompiled(m,independent)
        self.assertGreater(float(max(abs(duplicate.posterior_target()-base.posterior_target()))),1e-4)
        xor=np.empty((2,2,2))
        for w in itertools.product([0,1],repeat=3):xor[w]=.75 if w[0]^w[1]^w[2] else .25
        fallback=FactorWorlds([2]*3,[(tuple(range(3)),xor)])
        start=fallback.risk();fallback.update(1,0);self.assertAlmostEqual(start-fallback.risk(),0)
        fallback.update(2,0);self.assertAlmostEqual(start-fallback.risk(),.25)

    def test_exact_count_unchanged_path(self):
        m=latent_count_model(3,2,41)
        r=Release((1.,1.),(1.,1.),kind='exact')
        e=ProjectionCompiled(m,r);reference=Enumerated(m,r)
        np.testing.assert_allclose(e.posterior_target(),reference.posterior_target(),atol=1e-12)
        self.assertEqual(e.slopes,{})
        e.update(1,0);reference.update(1,0)
        np.testing.assert_allclose(e.posterior_target(),reference.posterior_target(),atol=1e-12)

    def test_deferred_updates_refuse_before_mutating_current_messages(self):
        e=ProjectionCompiled(multistate(2),Release((2.,2.),(1.,1.)))
        before=e.posterior_target().copy()
        buffers=[(id(n.poly),id(n.offset),id(n.logmass)) for n in e.nodes]
        with self.assertRaisesRegex(ValueError,'eager current messages'):
            e.update(1,1,defer=True)
        self.assertEqual(e.evidence,{})
        self.assertEqual(buffers,[(id(n.poly),id(n.offset),id(n.logmass)) for n in e.nodes])
        np.testing.assert_array_equal(e.posterior_target(),before)


if __name__ == '__main__':unittest.main()
