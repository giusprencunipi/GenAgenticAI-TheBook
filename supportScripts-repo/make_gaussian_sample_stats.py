#!/usr/bin/env python3
"""Recompute the many-samples column of the diffusion chapter's example on
sampling from the standard Gaussian, the one titled

    Five draws, and what they can and cannot tell

    python3 supportScripts-repo/make_gaussian_sample_stats.py

The example works five draws from N(0, I) in four dimensions by hand, and then
sets the resulting estimates of the mean, the variance and a covariance beside
the same estimates from many draws, to show that the properties of the
distribution only become visible once there are enough samples. The five
hand-worked vectors are printed in the book; this script produces the other
column: it draws N vectors in R^4 with a fixed seed and prints

    mean of x_1, mean of x_2          empirical means (theory: 0)
    variance of x_1                   empirical variance, 1/N (theory: 1)
    covariance of x_1 and x_2         empirical covariance, 1/N (theory: 0)

together with the largest absolute covariance over all six pairs of
dimensions, so that the one pair quoted is not a lucky choice. The estimators
divide by N, as the hand-worked five-sample column does, so the two columns are
the same computation at two sample sizes.

Requires numpy. Seeded, so a second run reproduces the first. Runs in well
under a second.
"""

import itertools

import numpy as np

N, D, SEED = 10_000, 4, 0


def main():
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((N, D))           # N independent draws of N(0, I)
    mean = x.mean(axis=0)
    centred = x - mean
    cov = centred.T @ centred / N              # 1/N, as in the book's column
    worst = max(abs(cov[i, j]) for i, j in itertools.combinations(range(D), 2))

    print("N = %d draws in R^%d, seed %d" % (N, D, SEED))
    print("  mean x_1            %+.3f" % mean[0])
    print("  mean x_2            %+.3f" % mean[1])
    print("  variance x_1        %.3f" % cov[0, 0])
    print("  covariance x_1,x_2  %+.3f" % cov[0, 1])
    print("  largest |covariance| over the six pairs  %.3f" % worst)


if __name__ == "__main__":
    main()
