from fractions import Fraction as F
import itertools
import unittest
import numpy as np
from disclosure.model import Release
from disclosure.adjoint_likelihood import AdjointLikelihood
from disclosure.adjoint_exact import exact_adjoint_path
from test_reverse_suffix import bundle_model,rational_worlds,rational_target

class NonplayerTests(unittest.TestCase):
    def test_complete_nonplayer_integration_and_retarget(self):
        m=bundle_model();lk=lambda s:F(1+(s[0]+2*s[1])%5,7)
        class Joint:
            observed,kind,identity=(0,0),'joint','alm-fixed-nonplayer-joint'
            def log_likelihood(self,c):return np.log((1+(c[...,0]+2*c[...,1])%5)/7)
        r=Joint();post=rational_worlds(m,lk);e=AdjointLikelihood(m,r,players=(1,2))
        for w,prob in post:
            if not prob:continue
            for o in ((1,2),(2,1)):
                ref=exact_adjoint_path(m,r,w,o,players=(1,2),likelihood=lk)
                for loss in ('zero_one','brier'):
                    out=e.evaluate_path(w,o,loss);evidence={}
                    for k in range(3):
                        target=rational_target(post,0,evidence,3);self.assertEqual(ref[k]['posterior'],target)
                        np.testing.assert_allclose(out.target_posteriors[k],list(map(float,target)),rtol=0,atol=3e-14)
                        if k<2:evidence[o[k]]=w[o[k]]
        changed=e.retarget(1);self.assertEqual(changed.players,(2,));self.assertIn(0,changed.fixed)
        fresh=AdjointLikelihood(changed.model,r,players=(2,))
        w=next(w for w,p in post if p)
        np.testing.assert_array_equal(changed.evaluate_path(w,(2,)).increments,fresh.evaluate_path(w,(2,)).increments)
        with self.assertRaises(ValueError):changed.evaluate_path(w,(1,2))

if __name__=='__main__':unittest.main()
