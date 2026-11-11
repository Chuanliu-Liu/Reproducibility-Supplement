"""Finite rational-box ALM reference, independent of packed implementation."""
from fractions import Fraction as F
import itertools
from .coordinates import UnimodularCertificate, contribution_box
from .reverse_suffix_exact import rational_risk
from .oracle import ImpossibleObservation


def exact_adjoint_path(model,release,world,order,*,players=None,matrix=None,likelihood=None,audit=None):
    q=len(release.observed);hcount=len(model.mixture);target=model.target
    coord=UnimodularCertificate.create(tuple(tuple(int(i==j) for j in range(q)) for i in range(q)) if matrix is None else matrix,q)
    c=[[coord.encode(v) for v in owner] for owner in model.contributions]
    p=[[[F(float(v)) for v in row] for row in owner] for owner in model.probs]
    players=tuple(i for i in range(len(p)) if i!=target) if players is None else tuple(players);order=tuple(order)
    if target in players or set(order)!=set(players) or len(order)!=len(players):raise ValueError('Original permutation required')
    fixed=[i for i in range(len(p)) if i!=target and i not in players]
    if likelihood is None:
        if release.kind!='exact':raise ValueError('Explicit rational full-joint likelihood required')
        likelihood=lambda s:F(s==tuple(release.observed))
    def domain(owners):
        box=contribution_box([c[i] for i in owners]);return itertools.product(*[range(a,b+1) for a,b in zip(box['low'],box['high'])])
    active=list(range(len(p)))
    table={s:[F(likelihood(coord.decode(s)))]*hcount for s in domain(active)}
    def eliminate(i):
        nonlocal table,active
        active.remove(i);out={}
        for t in domain(active):
            out[t]=[sum(p[i][h][v]*table[tuple(a+b for a,b in zip(t,f))][h] for v,f in enumerate(c[i])) for h in range(hcount)]
        table=out
    for i in fixed:eliminate(i)
    weights=[[F(float(v)) for v in model.mixture]];totals=[(0,)*q]
    for i in order:
        weights.append([a*p[i][h][world[i]] for h,a in enumerate(weights[-1])]);totals.append(tuple(a+b for a,b in zip(totals[-1],c[i][world[i]])))
    results=[None]*(len(order)+1)
    for k in range(len(order),-1,-1):
        mass=[sum(weights[k][h]*p[target][h][a]*table[tuple(x+y for x,y in zip(totals[k],f))][h] for h in range(hcount)) for a,f in enumerate(c[target])]
        if sum(mass)==0:raise ImpossibleObservation('Exact impossible observation')
        posterior=[v/sum(mass) for v in mass]
        results[k]=dict(posterior=posterior,mass=mass,risks={loss:rational_risk(posterior,model.target_labels,loss) for loss in ('zero_one','brier')})
        if audit:audit(k,table,active,results[k])
        if k:eliminate(order[k-1])
    return results
