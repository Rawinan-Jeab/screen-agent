"""Small, dependency-free statistics used in the analysis plan (section 6.7)."""
from __future__ import annotations

import math


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired booleans. Returns (b_only_a, only_b, p)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return only_a, only_b, min(1.0, 2 * p)


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def mcnemar_n(p10, p01, alpha=0.05, power=0.80):
    """Paired sample size (Connor 1987), the same formula as the Week 2 slide."""
    from statistics import NormalDist
    za = NormalDist().inv_cdf(1 - alpha / 2)
    zb = NormalDist().inv_cdf(power)
    pd, d = p10 + p01, p10 - p01
    return math.ceil((za * math.sqrt(pd) + zb * math.sqrt(pd - d * d)) ** 2 / (d * d))
