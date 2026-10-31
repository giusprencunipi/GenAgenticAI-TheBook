#!/usr/bin/env python3
"""Regenerate the forward-process image used by Chapter 6, in the figure
captioned "The forward process, run on a real MNIST digit" (fig:forward_real).

    forward_process.png     one real MNIST digit at nine points of the
                            1000-step forward process, under both the linear
                            and the cosine schedule

Nothing is trained here and nothing is learned: the forward process has no
parameters, so the whole figure is arithmetic on a real image. That is the
point the figure exists to make, and it is why this script is short.

    python3 supportScripts-repo/make_diffusion_figure.py

Two schedules are drawn, one per row: the linear schedule of the original
DDPM (T = 1000, beta from 1e-4 to 0.02) and the cosine schedule that replaced
it. Pixels are mapped to [-1, 1] the way the DDPM paper maps them, corrupted
in one shot with the closed form of the section "The Closed-Form Forward
Distribution" (sec:closed_form), and mapped back for
display. The same noise draw is reused at every step and in both rows, so the
grain is the same everywhere and only its weight changes: what the eye follows
is the digit fading, not the noise changing. Values outside [-1, 1] are
clipped when written out -- unavoidable, since a Gaussian is unbounded and a
PNG is not, and worth knowing when reading the last tiles.

The image is written as 8-bit greyscale, which is achromatic and so falls
outside the book's four-colour budget.

Requires numpy and pillow. Seeded, so it reproduces.
"""
import gzip, os, subprocess, sys, urllib.request
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".mnist-cache")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

T, BETA_MIN, BETA_MAX, COS_S = 1000, 1e-4, 0.02, 0.008
STEPS = [0, 50, 100, 200, 300, 450, 600, 800, 1000]
DIGIT, SEED, SCALE, GUTTER, ROWGAP = 3, 7, 4, 6, 10


def _download(url, path):
    """Fetch `url` to `path`, falling back to curl (see make_mnist_figures)."""
    try:
        urllib.request.urlretrieve(url, path)
        return
    except Exception as exc:                       # noqa: BLE001 -- any failure
        print("%% urllib failed (%s); trying curl" % exc, file=sys.stderr)
    if subprocess.call(["curl", "-sSL", "--fail", "-o", path, url]) != 0:
        raise SystemExit("could not download " + url)


def mnist():
    """Download (once) and decode the MNIST training images and labels."""
    os.makedirs(CACHE, exist_ok=True)
    out = []
    for name, header in (("train-images-idx3-ubyte.gz", 16),
                         ("train-labels-idx1-ubyte.gz", 8)):
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            _download(MIRROR + name, path)
        with gzip.open(path, "rb") as f:
            f.read(header)
            out.append(np.frombuffer(f.read(), dtype=np.uint8))
    return out[0].reshape(-1, 28, 28), out[1]


def alphabar_linear():
    """The cumulative product of 1 - beta_t for the linear DDPM schedule."""
    beta = np.linspace(BETA_MIN, BETA_MAX, T)
    return np.concatenate([[1.0], np.cumprod(1.0 - beta)])


def alphabar_cosine():
    """The cosine schedule, which is specified through alphabar directly."""
    t = np.arange(0, T + 1)
    f = np.cos(((t / T + COS_S) / (1 + COS_S)) * np.pi / 2) ** 2
    return f / f[0]


def main():
    images, labels = mnist()
    # The most legible three in the first hundred, chosen once and fixed.
    x0 = images[np.where(labels == DIGIT)[0][2]].astype(np.float64)
    x0 = x0 / 127.5 - 1.0                          # MNIST ink -> [-1, 1]

    rng = np.random.default_rng(SEED)
    eps = rng.standard_normal(x0.shape)            # one draw, reused throughout

    rows = []
    for name, ab in (("linear", alphabar_linear()),
                     ("cosine", alphabar_cosine())):
        tiles = []
        for t in STEPS:
            xt = np.sqrt(ab[t]) * x0 + np.sqrt(1.0 - ab[t]) * eps
            grey = np.clip((xt + 1.0) * 127.5, 0, 255).astype(np.uint8)
            tiles.append(255 - grey)               # dark ink on white paper
            print("%s t=%4d  alphabar=%.5f" % (name, t, ab[t]))
        rows.append(tiles)

    h = w = 28 * SCALE
    width = len(STEPS) * w + (len(STEPS) - 1) * GUTTER
    strip = np.full((2 * h + ROWGAP, width), 255, dtype=np.uint8)
    for r, tiles in enumerate(rows):
        for i, tile in enumerate(tiles):
            big = np.kron(tile, np.ones((SCALE, SCALE), dtype=np.uint8))
            y = r * (h + ROWGAP)
            strip[y:y + h, i * (w + GUTTER):i * (w + GUTTER) + w] = big

    out = os.path.join(HERE, "forward_process.png")
    Image.fromarray(strip, mode="L").save(out)
    print("wrote", out, strip.shape)


if __name__ == "__main__":
    main()
