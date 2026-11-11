"""Small finite-instance RSC reference: rational coefficients and outward exp.

Independent enumeration is supplied by the tests, not by this suffix reference.
Binary64 input probabilities are interpreted as their exact stored rationals.
The optional rational_likelihood adapter is explicit (e.g. RR of COUNT parity).
"""
from fractions import Fraction as F
from .exact_tree import exp_minus_bounds, CertificationBudgetExceeded
from .coordinates import UnimodularCertificate
from .oracle import ImpossibleObservation


def rational_risk(posterior, labels, loss):
    mass = {}
    for a, p in enumerate(posterior):
        mass[labels[a]] = mass.get(labels[a], F(0))+p
    if loss == 'zero_one':
        return 1-max(mass.values())
    if loss == 'brier':
        return (1-sum(p*p for p in mass.values()))/2
    raise ValueError(loss)


def exact_suffix_path(model, release, world, order, *, loss='zero_one', matrix=None,
                      players=None, rational_likelihood=None, bits=100, audit=None,
                      max_support=100000):
    q, hcount = model.contributions[0].shape[1], len(model.mixture)
    identity = tuple(tuple(int(i == j) for j in range(q)) for i in range(q))
    coord = UnimodularCertificate.create(identity if matrix is None else matrix, q)
    c = [[coord.encode(row) for row in owner] for owner in model.contributions]
    p = [[[F(float(v)) for v in row] for row in owner] for owner in model.probs]
    pi = [F(float(v)) for v in model.mixture]
    players = tuple(i for i in range(len(p)) if i != model.target) if players is None else tuple(players)
    order = tuple(order)
    if len(order) != len(players) or set(order) != set(players) or model.target in order:
        raise ValueError('Invalid original permutation')
    fixed = [i for i in range(len(p)) if i != model.target and i not in players]
    factors = []
    for owner in range(len(p)):
        f = [{} for _ in range(hcount)]
        for h in range(hcount):
            for v, s in enumerate(c[owner]):
                f[h][s] = f[h].get(s, F(0))+p[owner][h][v]
        factors.append(f)
    def add(table, owner):
        out = [{} for _ in range(hcount)]
        for h in range(hcount):
            for s, a in table[h].items():
                for f, b in factors[owner][h].items():
                    key = tuple(x+y for x, y in zip(s, f))
                    out[h][key] = out[h].get(key, F(0))+a*b
                    if len(out[h]) > max_support:
                        raise CertificationBudgetExceeded('Exact suffix support budget')
        return out
    table = [{tuple([0]*q): F(1)} for _ in range(hcount)]
    for i in fixed:
        table = add(table, i)
    # Explicit prefix products are correctly charged O(H n) retained metadata;
    # zeros revive without ever dividing by an observed probability.
    weights, totals = [pi], [tuple([0]*q)]
    for i in order:
        weights.append([a*p[i][h][int(world[i])] for h, a in enumerate(weights[-1])])
        totals.append(tuple(a+b for a, b in zip(totals[-1], c[i][int(world[i])])))
    results = [None]*(len(order)+1)
    for k in range(len(order), -1, -1):
        lower, upper = [F(0)]*model.domains[model.target], [F(0)]*model.domains[model.target]
        entries = []
        for h in range(hcount):
            for a, target_sum in enumerate(c[model.target]):
                for s, coefficient in table[h].items():
                    total = tuple(x+y+z for x, y, z in zip(totals[k], target_sum, s))
                    original = coord.decode(total)
                    w = weights[k][h]*p[model.target][h][a]*coefficient
                    if rational_likelihood is not None:
                        lk = F(rational_likelihood(original))
                        if lk < 0:
                            raise ValueError('Negative likelihood')
                        lower[a] += w*lk; upper[a] += w*lk
                    elif release.kind == 'exact':
                        if original == tuple(release.observed):
                            lower[a] += w; upper[a] += w
                    elif release.kind == 'laplace':
                        exponent = sum(abs(F(v)-F(float(y)))/F(float(b)) for v, y, b in zip(original, release.observed, release.scales))
                        if w:
                            entries.append((a, w, exponent))
                    else:
                        raise ValueError('An explicit rational joint-likelihood adapter is required')
        if entries:
            shift = min(e for _, _, e in entries)
            cache = {}
            for a, w, exponent in entries:
                exponent -= shift
                if exponent not in cache:
                    cache[exponent] = exp_minus_bounds(exponent, bits)
                lo, hi = cache[exponent]
                lower[a] += w*lo; upper[a] += w*hi
        if sum(upper) == 0:
            raise ImpossibleObservation('Exact zero normalization')
        if sum(lower) == 0:
            raise CertificationBudgetExceeded('No positive outward normalization bound')
        bounds = [(max(F(0), a/sum(upper)), min(F(1), b/sum(lower))) for a, b in zip(lower, upper)]
        label_bounds = {}
        for a, (lo, hi) in enumerate(bounds):
            label = model.target_labels[a]
            before = label_bounds.get(label, (F(0), F(0)))
            label_bounds[label] = (before[0]+lo, before[1]+hi)
        if loss == 'zero_one':
            risk = (max(F(0), 1-max(hi for _, hi in label_bounds.values())), min(F(1), 1-max(lo for lo, _ in label_bounds.values())))
        elif loss == 'brier':
            risk = (max(F(0), (1-sum(hi*hi for _, hi in label_bounds.values()))/2), min(F(1), (1-sum(lo*lo for lo, _ in label_bounds.values()))/2))
        else:
            raise ValueError(loss)
        results[k] = dict(posterior_bounds=bounds, risk_bounds=risk, target_mass_bounds=(lower, upper))
        if audit is not None:
            audit(k, table, totals[k], weights[k], results[k])
        if k:
            table = add(table, order[k-1])
    return results
