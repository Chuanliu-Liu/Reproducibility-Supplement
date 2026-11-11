import ctypes
from fractions import Fraction as F
import itertools
import os
from pathlib import Path
import unittest
import numpy as np
from disclosure.model import Model,Release,latent_count_model
from disclosure.adjoint_likelihood import AdjointLikelihood,directed_dyadic,fraction
from disclosure.adjoint_exact import exact_adjoint_path
from test_reverse_suffix import bundle_model,rational_worlds,rational_target,RRCountParity

LIB=os.environ.get('ALM_DIRECTED_LIBRARY')


class AdjointTests(unittest.TestCase):
    def test_all_worlds_permutations_whole_bundles_shared_joint(self):
        m=bundle_model();lk=lambda s:F(1+(s[0]+2*s[1])%5,7)
        class Joint:
            observed,scales,kind,identity=(0,0),(1,1),'joint','alm-coupled-joint'
            def log_likelihood(self,c):return np.log((1+(c[...,0]+2*c[...,1])%5)/7)
        posterior=rational_worlds(m,lk);negative=0;checked=0
        for matrix in (None,((1,0),(-1,1))):
            engine=AdjointLikelihood(m,Joint(),matrix=matrix,tile_cells=3)
            for w,prob in posterior:
                if not prob:continue
                for order in itertools.permutations((1,2,3)):
                    exact=exact_adjoint_path(m,Joint(),w,order,matrix=matrix,likelihood=lk)
                    for loss in ('zero_one','brier'):
                        out=engine.evaluate_path(w,order,loss);e={}
                        for k in range(4):
                            ref=rational_target(posterior,0,e,3)
                            self.assertEqual(ref,exact[k]['posterior'])
                            np.testing.assert_allclose(out.target_posteriors[k],list(map(float,ref)),rtol=0,atol=2e-14)
                            self.assertAlmostEqual(out.risks[k],float(exact[k]['risks'][loss]),places=13)
                            checked+=1
                            if k<3:e[order[k]]=w[order[k]]
                        negative+=int((out.increments<-1e-12).sum())
        self.assertGreater(negative,0);print('ALM exhaustive checked prefixes',checked,'negative increments',negative,flush=True)

    def test_n10_complete_world_oracle_H1_H2_arbitrary_target(self):
        for h in (1,2):
            m=latent_count_model(10,1,37,target=4)
            if h==1:m=Model([p[:1] for p in m.probs],[1.],m.contributions,4,version='alm-H1')
            r=Release((5,),(1,),kind='exact');post=rational_worlds(m,lambda s:F(s[0]==5))
            w=next(w for w,p in post if p);order=tuple(i for i in range(9,-1,-1) if i!=4)
            exact=exact_adjoint_path(m,r,w,order)
            engine=AdjointLikelihood(m,r)
            for loss in ('zero_one','brier'):
                out=engine.evaluate_path(w,order,loss);e={}
                for k in range(10):
                    ref=rational_target(post,4,e,2);self.assertEqual(ref,exact[k]['posterior'])
                    np.testing.assert_allclose(out.target_posteriors[k],list(map(float,ref)),rtol=0,atol=3e-14)
                    if k<9:e[order[k]]=w[order[k]]

    def test_hidden_nonplayers_target_truth_reset_and_invalidation(self):
        m=latent_count_model(5,2,9,target=2);r=Release((1.5,1.3),(1.1,.8),derived_answers=((0,2,1,4),))
        e=AdjointLikelihood(m,r,players=(0,1,4));w=[1,0,1,1,0];o=(4,1,0)
        a=e.evaluate_path(w,o);w[2]=0;w[3]=0;b=e.evaluate_path(w,o)
        np.testing.assert_array_equal(a.increments,b.increments)
        # Fixed prefix cannot use changed future player/target values.
        w[0]=0;c=e.evaluate_path(w,o)
        np.testing.assert_array_equal(a.target_posteriors[1],c.target_posteriors[1])
        r2=Release((2.5,1.3),(1.1,.8));new=e.change_release(r2)
        fresh=AdjointLikelihood(m,r2,players=(0,1,4))
        np.testing.assert_array_equal(new.evaluate_path(w,o).increments,fresh.evaluate_path(w,o).increments)
        m.probs[0][0,0]+=.001
        with self.assertRaises(ValueError):e.evaluate_path(w,o)

    def test_rr_xor_and_wrong_normalizer_domain_counterexamples(self):
        m=Model([[[.5,.5]]]*3,[1.],[[[0],[1]]]*3);r=RRCountParity()
        e=AdjointLikelihood(m,r);post=rational_worlds(m,r.rational_likelihood)
        for w,p in post:
            out=e.evaluate_path(w,(1,2));self.assertAlmostEqual(out.risks[0],.5);self.assertAlmostEqual(out.risks[1],.5);self.assertAlmostEqual(out.risks[2],.25)
        m=Model([[[.2,.8],[.8,.2]],[[.4,.6],[.3,.7]],[[.5,.5],[.5,.5]]],[.5,.5],[[[0],[1]]]*3)
        # Model accepts stored factors near normalization; their exact sums differ.
        m.probs[1][0]*=1.000005
        lk=lambda s:F(1,1+s[0]);r=Release((0,),(1,),kind='exact')
        exact=exact_adjoint_path(m,r,[0,0,0],(1,2),likelihood=lk)
        wrong=Model([p/p.sum(axis=1)[:,None] for p in m.probs],m.mixture,m.contributions)
        other=exact_adjoint_path(wrong,r,[0,0,0],(1,2),likelihood=lk)
        self.assertNotEqual(exact[0]['posterior'],other[0]['posterior'])
        # Incorrect signed offset addresses a different likelihood cell.
        signed=bundle_model();r=Release((.5,-.5),(1,1));good=AdjointLikelihood(signed,r);bad=AdjointLikelihood(signed,r)
        bad.base_low+=1
        with self.assertRaises(ValueError):bad.evaluate_path([0,0,1,0],(1,2,3))

    @unittest.skipUnless(LIB,'packed library required')
    def test_directed_contract_underflow_and_exact_operations(self):
        m=latent_count_model(2);r=Release((0,),(1,));e=AdjointLikelihood(m,r,certified=True,directed_library=LIB)
        operands=[(F(1,10),F(7,10)),(F(1,2**1074),F(1,2)),(F(1,2**1022),F(1,3)),(F(1),F(1,2**54)),(F(0),F(1))]
        for a,b in operands:
            x,y=float(a),float(b);exact=F(x)*F(y)
            for upper in (0,1):
                out=np.zeros(1);e.kernel(out,np.array([x]),1,1,1,0,0,y,upper)
                self.assertTrue(F(out[0])>=exact if upper else F(out[0])<=exact)
                self.assertEqual(e.lib.alm_round(),0)
        for bits in (64,128,192):
            for exponent in (-1200,-1074,-100,0,100):
                m=(1<<bits)+37
                for dtype in (np.float64,np.longdouble):
                    lo=directed_dyadic(m,exponent,dtype,False);hi=directed_dyadic(m,exponent,dtype,True);exact=F(m)*F(2)**exponent
                    self.assertLessEqual(fraction(lo),exact);self.assertGreaterEqual(fraction(hi),exact)

    @unittest.skipUnless(LIB,'packed library required')
    def test_enclosures_independent_rational_oracle_zero_revival_laplace(self):
        m=bundle_model();r=Release((-3,3),(1,1),kind='exact');post=rational_worlds(m,lambda s:F(s==(-3,3)))
        w=next(w for w,p in post if p)
        for dtype in ('float64','longdouble'):
            e=AdjointLikelihood(m,r,dtype=dtype,certified=True,directed_library=LIB,matrix=((1,0),(-1,1)))
            for loss in ('zero_one','brier'):
                out=e.evaluate_path(w,(1,2,3),loss);exact=exact_adjoint_path(m,r,w,(1,2,3),matrix=((1,0),(-1,1)))
                for k,record in enumerate(out.enclosures):
                    for p,(lo,hi) in zip(exact[k]['posterior'],record['posterior_bounds']):self.assertLessEqual(lo,p);self.assertGreaterEqual(hi,p)
                    lo,hi=record['risk_bounds'];self.assertLessEqual(lo,exact[k]['risks'][loss]);self.assertGreaterEqual(hi,exact[k]['risks'][loss])
        from disclosure.reverse_suffix_exact import exact_suffix_path
        m=latent_count_model(5,2,21);r=Release((1.25,1.5),(.8,1.2));w=[1,0,1,0,1];o=(4,3,2,1)
        for precision in (64,128,192):
            e=AdjointLikelihood(m,r,dtype='longdouble',certified=True,precision=precision,directed_library=LIB)
            out=e.evaluate_path(w,o,'brier');other=exact_suffix_path(m,r,w,o,loss='brier',bits=100)
            for a,b in zip(out.enclosures,other):
                self.assertLessEqual(max(a['risk_bounds'][0],b['risk_bounds'][0]),min(a['risk_bounds'][1],b['risk_bounds'][1]))


if __name__=='__main__':unittest.main()
