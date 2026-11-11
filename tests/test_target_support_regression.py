"""Target participates in the release even though it is not a player."""
import itertools
import math
import unittest
import numpy as np
from disclosure.model import Model, Release
from disclosure.absorption import AbsorptionCompiled, support_sign
from disclosure.projection import ProjectionCompiled


def enumerate_target(model, y, evidence):
    # Independent world/class enumeration, including the target in every sum.
    mass = np.zeros(len(model.target_labels))
    sums = []
    for world in itertools.product(*(range(d) for d in model.domains)):
        if any(world[i] != value for i, value in evidence.items()):
            continue
        q = sum(int(model.contributions[i][world[i], 0]) for i in range(len(world)))
        sums.append(q)
        prior = sum(float(model.mixture[h]) * math.prod(float(model.probs[i][h, x])
                    for i, x in enumerate(world)) for h in range(len(model.mixture)))
        mass[world[model.target]] += prior * math.exp(-abs(y-q))
    return min(sums), max(sums), mass / mass.sum()


class TargetSupportRegression(unittest.TestCase):
    def test_omitted_target_wrong_branch(self):
        m = Model([np.array([[.5,.5]]), np.array([[1.]])], np.array([1.]),
                  [np.array([[0],[1]]), np.array([[0]])])
        self.assertEqual(support_sign(0, 0, 0), 1)  # incorrect U-only bound
        for cls in (AbsorptionCompiled, ProjectionCompiled):
            e = cls(m, Release((0.,), (1.,)))
            self.assertEqual((int(e.lower[0]), int(e.upper[0])), (0, 1))
            self.assertEqual(e.slopes[0], -1)
            np.testing.assert_allclose(e.posterior_target(), [1/(1+math.exp(-1)), 1/(1+math.exp(1))], atol=1e-13)

    def test_all_prefixes_and_branches_against_world_enumeration(self):
        m = Model([np.array([[.2,.3,.5],[.5,.25,.25]]),
                   np.array([[.4,.6],[.7,.3]]), np.array([[.75,.25],[.2,.8]])],
                  np.array([.4,.6]),
                  [np.array([[-1],[1],[3]]), np.array([[0],[2]]), np.array([[-2],[1]])],
                  target_labels=(0,1,2))
        for y in (-5., 0., 2., 7.):
            for a,b in itertools.product((-1,0,1), repeat=2):
                ev = {i:v for i,v in ((1,a),(2,b)) if v >= 0}
                lo, hi, posterior = enumerate_target(m,y,ev)
                for cls in (AbsorptionCompiled, ProjectionCompiled):
                    e = cls(m, Release((y,), (1.,)), convolution_method='direct')
                    for i,v in ev.items(): e.update(i,v)
                    self.assertEqual((int(e.lower[0]),int(e.upper[0])), (lo,hi))
                    np.testing.assert_allclose(e.posterior_target(),posterior,atol=2e-12,rtol=0)
                    for c in e.certificates:
                        self.assertTrue(y <= int(c['lower']) or y >= int(c['upper']))

if __name__ == '__main__':
    unittest.main()
