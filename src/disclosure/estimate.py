import math
import time
import numpy as np


def hoeffding_radius(m, k, alpha=.05):
    if k == 0:
        return 0.
    if m <= 0 or not 0 < alpha < 1:
        raise ValueError('Invalid iid fixed-sample interval configuration')
    return math.sqrt(2*math.log(2*k/alpha)/m)


class PathBudgetExceeded(RuntimeError):
    def __init__(self, partial):
        super().__init__('Attribution time budget reached before completing this path')
        self.partial = partial


def path(engine, world, order, loss='zero_one', safe_prune=False, deadline=None):
    engine.reset()
    before = engine.risk(loss)
    initial = before
    inc = np.zeros(len(world))
    certificates = []
    for position, i in enumerate(order):
        if deadline is not None and time.perf_counter() >= deadline:
            raise PathBudgetExceeded({'world':list(map(int,world)), 'order':list(map(int,order)),
                'next_position':position,'increments':inc.tolist(),'risk_before_next':before,
                'initial_risk':initial,'evidence':{int(k):int(v) for k,v in engine.evidence.items()},
                'certificates':certificates})
        if i == engine.model.target:
            raise ValueError('Target cannot be a disclosure player')
        cert = engine.pruning_certificate(i) if safe_prune else None
        # Even a certified zero at this prefix is revealed for future synergy.
        if cert is not None and cert['safe_zero_current_state']:
            if hasattr(engine, 'update_certified_zero'):
                engine.update_certified_zero(int(i), int(world[i]), cert)
            else:
                engine.update(int(i), int(world[i]))
            after = before
            certificates.append(cert)
        else:
            engine.update(int(i),int(world[i]))
            after = engine.risk(loss)
        inc[i] = before-after  # Preserve negative realized increments.
        before = after
    if not np.isclose(inc.sum(), initial-before, atol=1e-10):
        raise ArithmeticError('Path failed efficiency')
    return inc, certificates


def draws(engine, m, seed):
    rng = np.random.default_rng(seed)
    players = np.array([i for i in range(len(engine.model.probs)) if i != engine.model.target])
    engine.reset()
    return [(engine.sample_world(rng), rng.permutation(players)) for _ in range(m)]


def estimate(engine, common_draws, loss='zero_one', safe_prune=False, alpha=.05):
    n = len(engine.model.probs)
    total = np.zeros(n); squares = np.zeros(n)
    negative = 0; certificates = []
    for w, order in common_draws:
        inc, cert = path(engine,w,order,loss,safe_prune)
        total += inc; squares += inc*inc
        negative += int((inc < -1e-12).sum())
        certificates.extend(cert)
    m = len(common_draws)
    return {'mean':(total/m).tolist(),'sum':total.tolist(),'sum_squares':squares.tolist(),
            'paths':m,'negative_realized_increments':negative,'alpha':alpha,
            'simultaneous_iid_hoeffding_radius':hoeffding_radius(m,n-1,alpha),
            'replication_unit':'independent complete posterior world and independent uniform permutation',
            'pruning_certificates':certificates}
