"""One-time input and integer-range checks for finite owner compilers.

The mathematical scalar representation uses Python-integer support bounds.
The ordinary NumPy tree additionally needs a checked integer/float range.
These checks perform no inference and allocate no sum-state tables.
"""
from dataclasses import dataclass
import math
import operator
import numpy as np
from .coordinates import UnimodularCertificate


class UnsafeSupportError(ValueError):
    """The ordinary numeric tree cannot safely represent these coordinates."""


@dataclass(frozen=True)
class FactorInputs:
    n: int
    h: int
    q: int
    domains: tuple
    labels: tuple
    mins: tuple
    maxs: tuple
    lower: tuple
    upper: tuple


def _integer(value, description):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(description + ' must be an integer, not a boolean')
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(description + ' must be an integer') from exc


def validate_factor_inputs(model, release):
    """Validate received factors without coercing a noninteger contribution."""
    if release.kind not in ('laplace', 'exact'):
        raise ValueError('Independent Laplace or exact release required')
    q = len(release.observed)
    if q not in (1, 2) or len(release.scales) != q:
        raise ValueError('Matching q=1 or q=2 release required')
    if any(isinstance(v, np.floating) and v.dtype.itemsize > 8
           for v in (*release.observed, *release.scales)):
        raise ValueError('Extended-precision release parameters need a separate numeric backend')
    try:
        observed = tuple(float(y) for y in release.observed)
        scales = tuple(float(b) for b in release.scales)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Finite numeric release parameters required') from exc
    if not all(math.isfinite(y) for y in observed):
        raise ValueError('Finite observations required')
    if not all(math.isfinite(b) and b > 0 for b in scales):
        raise ValueError('Finite positive release scales required')
    mixture = np.asarray(model.mixture)
    if (mixture.ndim != 1 or not len(mixture)
            or not np.all(np.isfinite(mixture))
            or np.any(mixture < 0)
            or not np.isclose(mixture.sum(), 1.0)):
        raise ValueError('Finite normalized nonnegative class weights required')
    n, h = len(model.probs), len(mixture)
    if n == 0 or len(model.contributions) != n:
        raise ValueError('Matching nonempty owner factors required')
    target = _integer(model.target, 'Target owner')
    if not 0 <= target < n:
        raise ValueError('Target owner outside model')
    domains, mins, maxs = [], [], []
    for i, (probability, contribution) in enumerate(zip(model.probs, model.contributions)):
        p, c = np.asarray(probability), np.asarray(contribution)
        if (p.ndim != 2 or c.ndim != 2 or c.shape[1] != q
                or c.shape[0] == 0 or p.shape != (h, c.shape[0])):
            raise ValueError('Invalid complete owner factor shape')
        if (not np.all(np.isfinite(p)) or np.any(p < 0)
                or not np.allclose(p.sum(axis=1), 1.0)):
            raise ValueError('Finite normalized nonnegative owner probabilities required')
        rows = [tuple(_integer(v, 'Owner contribution') for v in row) for row in c]
        domains.append(len(rows))
        mins.append(tuple(min(row[d] for row in rows) for d in range(q)))
        maxs.append(tuple(max(row[d] for row in rows) for d in range(q)))
    labels = tuple(_integer(v, 'Target label') for v in model.target_labels)
    if len(labels) != domains[target] or any(v < 0 for v in labels):
        raise ValueError('One nonnegative integer label per target state required')
    # Python sums, including their intermediate values, never wrap at int64.
    lower = tuple(sum(row[d] for row in mins) for d in range(q))
    upper = tuple(sum(row[d] for row in maxs) for d in range(q))
    return FactorInputs(n, h, q, tuple(domains), labels,
                        tuple(mins), tuple(maxs), lower, upper)


def guard_compiled_extent(model, release, options=None, validated=None):
    """Reject unsafe ordinary-tree arithmetic before encoding or allocation.

All candidate coordinate systems are checked because the existing compiler
constructs each candidate before choosing one. Absolute owner bounds protect
every subtree and growing prefix, including cancellation between owners.
The 2**52-1 limit also keeps integer sum-grid arithmetic exactly representable
in binary64; it is a backend admission bound, not a new model assumption.
"""
    data = validated if validated is not None else validate_factor_inputs(model, release)
    options = {} if options is None else options
    q = data.q
    limit = (1 << 52) - 1
    identity = tuple(tuple(int(i == j) for j in range(q)) for i in range(q))
    matrices = [identity]
    if options.get('query_matrix') is not None:
        matrices.append(options['query_matrix'])
    elif q == 2 and options.get('query_basis', True):
        matrices.extend((((1, 0), (-1, 1)), ((1, -1), (0, 1))))
    for matrix in matrices:
        certificate = UnimodularCertificate.create(matrix, q)
        for row in (*certificate.matrix, *certificate.inverse):
            if any(abs(v) > limit for v in row):
                raise UnsafeSupportError('Coordinate matrix exceeds checked numeric range')
        absolute = [0] * q
        for contribution in model.contributions:
            owner_absolute = [0] * q
            for row in contribution:
                encoded = certificate.encode(tuple(int(v) for v in row))
                for d, value in enumerate(encoded):
                    owner_absolute[d] = max(owner_absolute[d], abs(value))
            for d in range(q):
                absolute[d] += owner_absolute[d]
        if any(v > limit for v in absolute):
            raise UnsafeSupportError('Owner/subtree support exceeds checked numeric range')
        for row in certificate.inverse:
            if sum(abs(a) * b for a, b in zip(row, absolute)) > limit:
                raise UnsafeSupportError('Decoded sum support exceeds checked numeric range')
        try:
            transformed_y = [sum(float(a) * float(y) for a, y in zip(row, release.observed))
                             for row in certificate.matrix]
            slopes = [sum(abs(float(certificate.inverse[r][d])) / float(release.scales[r])
                          for r in range(q)) for d in range(q)]
        except (OverflowError, ZeroDivisionError) as exc:
            raise UnsafeSupportError('Release arithmetic exceeds numeric range') from exc
        if not all(math.isfinite(v) for v in (*transformed_y, *slopes)):
            raise UnsafeSupportError('Release arithmetic exceeds numeric range')
    return data
