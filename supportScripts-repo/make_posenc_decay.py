#!/usr/bin/env python3
"""Recompute how far the similarity of sinusoidal positional encodings decays.

    python3 supportScripts-repo/make_posenc_decay.py

The Remark on why nearby positions score high, in the Transformer chapter,
quotes two numbers for each model width d_model: the offset up to which the dot
product p_i . p_{i+delta} falls strictly, and the worst inversion within the
first 2048 offsets, that is the largest amount by which a more distant position
outscores a nearer one, as a percentage of the maximum d_model/2. Every one of
those figures is printed by this script.

The dot product is computed from the exact identity

    p_i . p_{i+delta} = sum_k cos(omega_k delta),  omega_k = 10000^(-2k/d_model),

which depends on the offset delta alone. Every pair of offsets in the range is
compared, not a sample: the worst inversions sit around delta = 1400-1900, and a
scan restricted to small offsets misses them.

No randomness and no dependencies beyond the standard library.
"""

import math

WIDTHS = (128, 256, 512, 768, 1024, 2048, 4096, 12288)
SPAN = 2048


def dot(delta, d, base=10000.0):
    """p_i . p_{i+delta} = sum_k cos(omega_k delta)."""
    return sum(math.cos(delta / base ** (2 * k / d)) for k in range(d // 2))


def survey(d, span=SPAN):
    v = [dot(x, d) for x in range(span + 1)]
    mono = next((x - 1 for x in range(1, span + 1) if v[x] > v[x - 1]), span)
    # Worst inversion. For each b, the largest violation involving b is the one
    # against the smallest value seen before it, so a single pass covers every
    # pair rather than a sample of them.
    best, arg, running, argmin = 0.0, (0, 0), v[0], 0
    for b in range(1, span + 1):
        if v[b] - running > best:
            best, arg = v[b] - running, (argmin, b)
        if v[b] < running:
            running, argmin = v[b], b
    return mono, 100 * best / v[0], arg


if __name__ == "__main__":
    print(f"{'d_model':>7} {'monotone to':>11} {'worst inversion':>15}  attained at")
    for d in WIDTHS:
        mono, pct, (a, b) = survey(d)
        print(f"{d:>7} {mono:>11} {pct:>14.3f}%  delta={b} over delta={a}")
