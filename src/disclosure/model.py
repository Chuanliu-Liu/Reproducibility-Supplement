"""Explicit finite owner-bundle latent-class prior and aggregate mechanism."""
from dataclasses import dataclass
import hashlib
import json
import operator
import numpy as np
from scipy.special import logsumexp


def _integer_contributions(raw):
    """Preserve exact input integers before the int64 storage conversion."""
    array = np.asarray(raw, dtype=object)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("Nonempty integer contribution matrix required")
    values = []
    for value in array.flat:
        try:
            if isinstance(value, (bool, np.bool_)):
                raise TypeError("Boolean contribution")
            integer = operator.index(value)
        except TypeError as exc:
            raise ValueError("Owner contributions must be exact integers") from exc
        if not -(1 << 63) <= integer < (1 << 63):
            raise ValueError("Owner contribution outside signed int64 storage range")
        values.append(integer)
    return np.asarray(values, dtype=np.int64).reshape(array.shape)


@dataclass
class Model:
    # A bundle state is disclosed as one player, even when containing many variables.
    probs: list  # Each owner: [latent classes, bundle states].
    mixture: np.ndarray
    contributions: list  # Each owner: [bundle states, query dimensions], integers.
    target: int = 0
    target_labels: tuple = (0, 1)
    version: str = "v1"
    owners: tuple = ()

    def __post_init__(self):
        self.probs = [np.asarray(p, dtype=float) for p in self.probs]
        self.mixture = np.asarray(self.mixture, dtype=float)
        self.contributions = [_integer_contributions(c) for c in self.contributions]
        if not np.isclose(self.mixture.sum(), 1) or (self.mixture < 0).any():
            raise ValueError("Invalid latent mixture")
        if not (len(self.probs) == len(self.contributions) and 0 <= self.target < len(self.probs)):
            raise ValueError("Invalid owner model")
        q = self.contributions[0].shape[1]
        for p, c in zip(self.probs, self.contributions):
            if p.shape != (len(self.mixture), len(c)) or c.shape[1] != q:
                raise ValueError("Invalid owner-bundle domain")
            if (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
                raise ValueError("Invalid bundle probability")
        if len(self.target_labels) != self.probs[self.target].shape[1]:
            raise ValueError("Missing target projection")
        if self.owners and len(set(self.owners)) != len(self.probs):
            raise ValueError("Owner identity splitting/duplication")

    @property
    def domains(self):
        return [p.shape[1] for p in self.probs]

    @property
    def identity(self):
        x = {"probs": [p.tolist() for p in self.probs], "mixture": self.mixture.tolist(),
             "contributions": [c.tolist() for c in self.contributions],
             "target": self.target, "labels": self.target_labels, "version": self.version,
             "owners": self.owners}
        return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

    def log_prior(self, w):
        with np.errstate(divide="ignore"):
            a = np.log(self.mixture)
            for p, v in zip(self.probs, w):
                a = a + np.log(p[:, v])
        return float(logsumexp(a))

    def counts(self, w):
        return np.sum([c[v] for c, v in zip(self.contributions, w)], axis=0)


@dataclass
class Release:
    observed: tuple
    scales: tuple
    kind: str = "laplace"
    release_id: str = "internal-synthetic"
    # Postprocessing of a shared synopsis is a support constraint, not a new likelihood.
    derived_answers: tuple = ()  # (dimension, multiplier, intercept, observed value)

    def __post_init__(self):
        if len(self.observed) != len(self.scales):
            raise ValueError("Release dimension mismatch")
        if any(s <= 0 for s in self.scales):
            raise ValueError("Strictly positive calibrated scales required")
        for d, a, b, y in self.derived_answers:
            if not np.isclose(a*self.observed[d]+b, y, rtol=0, atol=1e-12):
                raise ValueError("Impossible shared-synopsis derived answers")

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.__dict__, sort_keys=True).encode()).hexdigest()

    def log_likelihood(self, counts):
        c = np.asarray(counts)
        y = np.asarray(self.observed)
        s = np.asarray(self.scales)
        if self.kind == "laplace":
            return np.sum(-np.abs(c-y)/s - np.log(2*s), axis=-1)
        if self.kind == "exact":
            return np.where(np.all(c == y, axis=-1), 0., -np.inf)
        raise ValueError(self.kind)


def latent_count_model(n, q=1, seed=19, strength=.18, target=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(.25, .65, n)
    probs = [np.array([[1-a, a], [1-b, b]])
             for a, b in zip(np.clip(base-strength, .02, .98), np.clip(base+strength, .02, .98))]
    weights = np.ones((n, q), dtype=int)
    if q > 1:
        weights[:, 1] = np.arange(n) % 3 != 0  # Overlapping, shared-owner queries.
    contrib = [np.array([np.zeros(q, dtype=int), w]) for w in weights]
    return Model(probs, np.array([.45, .55]), contrib, target)
