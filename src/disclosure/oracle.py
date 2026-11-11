"""Independent scalar oracle. No dependency on the compiled engine or NumPy."""
import itertools
import math


class ImpossibleObservation(ValueError):
    pass


def normalize_worlds(domains, log_prior, log_likelihood):
    worlds = list(itertools.product(*[range(d) for d in domains]))
    logs = [log_prior(w) + log_likelihood(w) for w in worlds]
    high = max(logs)
    if high == -math.inf:
        raise ImpossibleObservation("Observed release has zero model support")
    raw = [math.exp(x - high) for x in logs]
    z = math.fsum(raw)
    return [(w, p / z) for w, p in zip(worlds, raw)], high + math.log(z)


def bayes_risk(mass, loss="zero_one"):
    z = math.fsum(mass.values())
    if z == 0:
        raise ImpossibleObservation("Disclosure has zero posterior support")
    p = [v / z for v in mass.values()]
    if loss == "zero_one":
        return 1 - max(p)
    if loss == "brier":
        # Half of multiclass Brier sum; binary equals (a-x)^2, range [0,1].
        return (1 - math.fsum(v*v for v in p)) / 2
    raise ValueError(loss)


def coalition_risk(posterior, target, coalition, labels=None, loss="zero_one"):
    groups = {}
    for w, p in posterior:
        key = tuple(w[i] for i in coalition)
        t = w[target] if labels is None else labels[w[target]]
        mass = groups.setdefault(key, {})
        mass[t] = mass.get(t, 0.) + p
    return math.fsum(math.fsum(m.values()) * bayes_risk(m, loss) for m in groups.values())


def exact_game(posterior, target, labels=None, loss="zero_one"):
    n = len(posterior[0][0])
    players = [i for i in range(n) if i != target]
    risks = {}
    for k in range(len(players)+1):
        for s in itertools.combinations(players, k):
            risks[s] = coalition_risk(posterior, target, s, labels, loss)
    m = len(players)
    phi = {}
    for i in players:
        phi[i] = math.fsum(
            (risks[s] - risks[tuple(sorted((*s, i)))]) /
            (m * math.comb(m-1, len(s)))
            for s in risks if i not in s)
    return {"risks": risks, "phi": phi,
            "efficiency_residual": math.fsum(phi.values()) -
            (risks[()] - risks[tuple(players)])}


def realized_risk(posterior, target, evidence, labels=None, loss="zero_one"):
    mass = {}
    for w, p in posterior:
        if all(w[i] == v for i, v in evidence.items()):
            t = w[target] if labels is None else labels[w[target]]
            mass[t] = mass.get(t, 0.) + p
    return bayes_risk(mass, loss)
