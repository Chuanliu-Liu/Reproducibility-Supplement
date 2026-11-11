from fractions import Fraction as F
import os
import unittest
import numpy as np
from disclosure.model import Release,latent_count_model
from disclosure.adjoint_likelihood import AdjointLikelihood,fraction

class DirectedContract(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('ALM_DIRECTED_LIBRARY'),'directed library required')
    def test_nonzero_accumulators_two_mantissas_and_subnormals(self):
        rng=np.random.default_rng(610007);count=0
        for dtype in ('float64','longdouble'):
            e=AdjointLikelihood(latent_count_model(2),Release((0,),(1,)),dtype=dtype,certified=True,directed_library=os.environ['ALM_DIRECTED_LIBRARY'])
            typ=e.dtype;small=np.finfo(typ).smallest_subnormal
            triples=[(typ(1),typ(1),.1),(typ(1),np.nextafter(typ(0),typ(1)),.5),(typ(0),small,.5),(small,small,.5),(typ(1),np.nextafter(typ(1),typ(2)),.7)]
            for _ in range(256):
                a,b=rng.uniform(0,1,2);triples.append((np.ldexp(typ(a),int(rng.integers(-1000,1000))),np.ldexp(typ(b),int(rng.integers(-1000,1000))),float(rng.uniform(0,1))))
            for a,b,p in triples:
                exact=fraction(a)+fraction(b)*F(p)
                for upper in (0,1):
                    out=np.array([a],dtype=typ);source=np.array([b],dtype=typ)
                    self.assertEqual(e.kernel(out,source,1,1,1,0,0,p,upper),0)
                    self.assertTrue(fraction(out[0])>=exact if upper else fraction(out[0])<=exact)
                    self.assertEqual(e.lib.alm_round(),0);count+=1
        print('ALM exact directed product-plus-add checks',count,flush=True)

if __name__=='__main__':unittest.main()
