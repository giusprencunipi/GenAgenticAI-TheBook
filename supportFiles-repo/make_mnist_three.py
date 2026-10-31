#!/usr/bin/env python3
"""Write mnist_three.png, the single handwritten 3 that Chapter 3 introduces in
the section "Example: One Digit, Start to Finish" (fig:mnist_three) and that
Chapter 4 picks up again.

    python3 supportFiles-repo/make_mnist_three.py

The digit is the first 3 of the MNIST training set, taken as it is: no
smoothing, no resampling. Each of the 28x28 pixels is drawn as a square block
with a thin grey rule around it, so that the figure shows what the text says,
a grid of 784 numbers. The image is 8-bit greyscale, dark ink on white like the
other MNIST figures, and so falls outside the four-colour budget.

Requires numpy and pillow; MNIST is downloaded once and cached, as in
make_mnist_figures.py.
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mnist_figures import mnist           # same loader, same cache

CELL, RULE = 16, 1                             # pixels per MNIST pixel, rule width


def main():
    X, y = mnist()
    idx = int(np.flatnonzero(y == 3)[0])
    digit = 255 - X[idx].reshape(28, 28)       # dark ink on white
    side = 28 * CELL + RULE
    img = np.full((side, side), 200, dtype=np.uint8)   # the grey rules
    for r in range(28):
        for c in range(28):
            y0, x0 = r * CELL + RULE, c * CELL + RULE
            img[y0:y0 + CELL - RULE, x0:x0 + CELL - RULE] = digit[r, c]
    Image.fromarray(img, mode="L").save("mnist_three.png", dpi=(600, 600))
    print("mnist_three.png: training image %d, label %d" % (idx, y[idx]))


if __name__ == "__main__":
    main()
