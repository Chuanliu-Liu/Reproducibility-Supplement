"""Public integer-coordinate certificates for q=1,2; no likelihood change.

A unimodular matrix certifies a bijection on Z^q and hence on every attainable
sum support. Sparse support cardinality is invariant; only its bounding box may
shrink. Selection must use public contributions, not hidden bundle states.
"""
from dataclasses import dataclass
import math
import operator


def _integer(value):
    if isinstance(value, bool):
        raise ValueError('Boolean coordinate is not an integer certificate input')
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError('Integer certificate input required') from exc


def transform(matrix, vector):
    vector = tuple(_integer(v) for v in vector)
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError('Coordinate dimension mismatch')
    return tuple(sum(_integer(a)*b for a, b in zip(row, vector)) for row in matrix)


@dataclass(frozen=True)
class UnimodularCertificate:
    matrix: tuple
    inverse: tuple
    determinant: int

    @classmethod
    def create(cls, matrix, q=None):
        matrix = tuple(tuple(_integer(v) for v in row) for row in matrix)
        dimension = len(matrix)
        if dimension not in (1, 2) or any(len(row) != dimension for row in matrix):
            raise ValueError('Only q<=2 square integer transforms are supported')
        if q is not None and dimension != q:
            raise ValueError('Transform/release dimension mismatch')
        if dimension == 1:
            det = matrix[0][0]
            inverse = matrix
        else:
            (a, b), (c, d) = matrix
            det = a*d-b*c
            inverse = None if det == 0 else ((d//det, -b//det), (-c//det, a//det))
        if det not in (-1, 1):
            raise ValueError('A unimodular transform with determinant +/-1 is required')
        identity = tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension))
        for left, right in [(matrix, inverse), (inverse, matrix)]:
            product = tuple(tuple(sum(left[i][k]*right[k][j] for k in range(dimension))
                                  for j in range(dimension)) for i in range(dimension))
            if product != identity:
                raise ValueError('Exact inverse identity failed')
        return cls(matrix, inverse, det)

    def encode(self, vector):
        return transform(self.matrix, vector)

    def decode(self, vector):
        return transform(self.inverse, vector)

    def as_dict(self):
        return dict(matrix=self.matrix, inverse=self.inverse, determinant=self.determinant,
                    dimension=len(self.matrix), certificate='U*inverse= inverse*U=I in integer arithmetic',
                    bijection_scope='all Z^q; restriction to every attainable support',
                    release_evaluation='original answer coordinates via inverse; joint likelihood retained')


def integer_inverse(matrix):
    return UnimodularCertificate.create(matrix).inverse


def contribution_box(contributions):
    """Exact Minkowski bounding box; no exponential support enumeration."""
    q = len(contributions[0][0])
    low = tuple(sum(min(int(row[d]) for row in c) for c in contributions) for d in range(q))
    high = tuple(sum(max(int(row[d]) for row in c) for c in contributions) for d in range(q))
    widths = tuple(b-a+1 for a, b in zip(low, high))
    return dict(low=low, high=high, widths=widths, volume=math.prod(widths))


def support_geometry(support):
    support = {tuple(_integer(v) for v in row) for row in support}
    if not support:
        return dict(cardinality=0, widths=(), bounding_volume=0)
    q = len(next(iter(support)))
    low = tuple(min(s[d] for s in support) for d in range(q))
    high = tuple(max(s[d] for s in support) for d in range(q))
    widths = tuple(b-a+1 for a, b in zip(low, high))
    return dict(cardinality=len(support), low=low, high=high, widths=widths,
                bounding_volume=math.prod(widths))
