"""Same-game weighted-counting decision DAG adaptation, not database Shapley.

Expands owner choices through an ordered deterministic decision DAG, merging
identical (owner index, latent class, accumulated query sum, target label) states.
Evidence indicators select whole-owner branches. Terminal weights use the actual
joint release likelihood. This reimplements arithmetic-circuit probability
evaluation (Darwiche UAI2000 Definition1; probabilistic-lineage counting in
Deutch et al. facts-query primary paper), then applies the specified Bayes-risk
game. Boolean query-value Shapley reductions are not imported into this game.
All arithmetic here is floating; state budget and full reevaluation are counted.
"""
import math
import time
import numpy as np
from .oracle import ImpossibleObservation


class LineageWMC:
    def __init__(self, model, release, max_nodes=250_000):
        start = time.perf_counter()
        self.model, self.release = model, release
        self.evidence, self.nodes, self.memo = {}, [], {}
        self.counters = dict(evaluations=0, edge_products=0, updates=0)
        n = len(model.probs)
        labels = max(model.target_labels)+1
        def build(i, h, sums, label):
            key = (i, h, sums, label)
            if key in self.memo:return self.memo[key]
            if len(self.memo) >= max_nodes:
                raise MemoryError(f'Lineage counting DAG exceeds {max_nodes} distinct states')
            # Reserve before recursion so the admission check includes ancestors.
            self.memo[key] = None
            edges = []
            if i < n:
                for value in range(model.domains[i]):
                    probability = float(model.probs[i][h, value])
                    if probability == 0:continue
                    answer = tuple(int(a+b) for a,b in zip(sums, model.contributions[i][value]))
                    target = model.target_labels[value] if i == model.target else label
                    child = build(i+1, h, answer, target)
                    edges.append((value, probability, child))
            idx = len(self.nodes)
            self.nodes.append(dict(owner=i, label=label, sums=sums, edges=edges))
            self.memo[key] = idx
            return idx
        self.roots = [build(0, h, tuple([0]*len(release.observed)), -1) for h in range(len(model.mixture))]
        terminals = [node for node in self.nodes if node['owner'] == n]
        logs = np.array([float(release.log_likelihood(np.array(node['sums']))) for node in terminals])
        shift = float(logs.max())
        if not math.isfinite(shift):raise ImpossibleObservation('No possible lineage release')
        for node, logweight in zip(terminals, logs):
            terminal = np.zeros(labels)
            terminal[node['label']] = math.exp(logweight-shift)
            node['terminal'] = terminal
        self.compile_seconds = time.perf_counter()-start

    def update(self, owner, value):
        if not 0 <= value < self.model.domains[owner]:raise ValueError('Invalid bundle state')
        self.evidence[owner] = value
        self.counters['updates'] += 1

    def reset(self):self.evidence.clear()

    def posterior_target(self):
        results = []
        labels = max(self.model.target_labels)+1
        for node in self.nodes:
            if not node['edges']:
                value = node.get('terminal', np.zeros(labels))
            else:
                value = np.zeros(labels)
                observed = self.evidence.get(node['owner'])
                for state, probability, child in node['edges']:
                    if observed is None or observed == state:
                        value += probability*results[child]
                        self.counters['edge_products'] += 1
            results.append(value)
            self.counters['evaluations'] += 1
        p = sum(w*results[root] for w,root in zip(self.model.mixture, self.roots))
        if p.sum() <= 0:raise ImpossibleObservation('Lineage evidence has zero/numerically lost mass')
        return p/p.sum()

    def risk(self, loss='zero_one'):
        p = self.posterior_target()
        if loss == 'zero_one':return float(1-p.max())
        if loss == 'brier':return float((1-np.square(p).sum())/2)
        raise ValueError(loss)
