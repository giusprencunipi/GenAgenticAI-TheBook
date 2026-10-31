#!/usr/bin/env python3
"""Compute the points plotted in the figure "A chain of invertible
transformations" of Chapter 4 (fig:flow_chain).

    python3 supportScripts-repo/make_flow_chain.py

300 points are drawn from the two-dimensional standard Gaussian and pushed
through three hand-written invertible maps. Each map changes one coordinate as
a function of the other, in the spirit of the coupling layers described in the
same section, so each is invertible in closed form and its Jacobian is
triangular:

    f1: (a, b) -> (a, 0.6 b + 0.5 a^2 - 1)
    f2: (a, b) -> (0.6 a + 1.6 tanh(1.5 b), b)
    f3: (a, b) -> (a, b - 0.4 a^2)

The points are grouped by the quadrant they start in, and the script prints,
for each of the four stages z0, z1, z2, x = z3, one pgfplots coordinate list
per group, ready to paste into vae.tex and it/vae.tex. Nothing is drawn here:
the figure is typeset by pgfplots so that it uses the book's fonts and inks.

Requires numpy only. Seeded, so it reproduces.
"""
import numpy as np

N, SEED = 300, 0


def f1(z):
    a, b = z[:, 0], z[:, 1]
    return np.c_[a, 0.6 * b + 0.5 * a ** 2 - 1.0]


def f2(z):
    a, b = z[:, 0], z[:, 1]
    return np.c_[0.6 * a + 1.6 * np.tanh(1.5 * b), b]


def f3(z):
    a, b = z[:, 0], z[:, 1]
    return np.c_[a, b - 0.4 * a ** 2]


def main():
    z = np.random.default_rng(SEED).standard_normal((N, 2))
    quadrant = (z[:, 0] > 0).astype(int) + 2 * (z[:, 1] > 0).astype(int)
    stages = [z]
    for f in (f1, f2, f3):
        stages.append(f(stages[-1]))
    for k, s in enumerate(stages):
        print("%% stage %d" % k)
        for q in range(4):
            pts = s[quadrant == q]
            print("coordinates {" + " ".join("(%.2f,%.2f)" % tuple(p)
                                            for p in pts) + "};")


if __name__ == "__main__":
    main()
