#!/usr/bin/env python3
"""Reproduce the table in Chapter 3, Section "Practical Considerations",
subsection "Choosing the Bottleneck Dimension L": the reconstruction error of
the chapter's autoencoder on MNIST for several widths of the bottleneck.

    python3 supportFiles-repo/make_ae_bottleneck_numbers.py

The book's table has one row per value of L. For each row this script trains
the chapter's autoencoder, 784-512-256-L-256-512-784 with ReLU hidden layers,
a linear code layer and a sigmoid output, on the 60,000 MNIST training images
with the MSE loss, and reports the reconstruction error on the 10,000 test
images, which the network never sees during training:

    L          width of the code
    ratio      784 / L, the compression ratio printed in the book
    test MSE   mean squared error per pixel, pixels scaled to [0, 1]
    PSNR       the same error as a peak signal-to-noise ratio in decibels,
               10 log10(1 / MSE), the usual way of reading reconstruction
               quality: higher is better, and each 3 dB halves the error

Everything else is held fixed (optimiser, learning rate, batch size, number
of epochs, seed), so that the rows differ only in L. The book's column
"Behaviour" is a reading of these numbers; if the book and this script
disagree, the script is right.

Requires torch and numpy. Seeded, so a second run reproduces the first. Runs
on the CPU in about eight minutes, a minute and a half per row.
"""
import os, sys, time
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ae_pretraining_numbers import mnist, Autoencoder   # same network

WIDTHS = (2, 10, 20, 64, 128)
EPOCHS, BATCH, LR, SEED = 30, 128, 1e-3, 0


def train(X, L):
    torch.manual_seed(SEED)
    ae = Autoencoder(z=L)
    opt = torch.optim.Adam(ae.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(X), generator=g).split(BATCH):
            x = X[idx]
            loss = F.mse_loss(ae(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
    return ae


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    Xtr, _ = mnist("train")
    Xte, _ = mnist("test")
    rows = []
    for L in WIDTHS:
        t0 = time.time()
        ae = train(Xtr, L)
        with torch.no_grad():
            mse = F.mse_loss(ae(Xte), Xte).item()
        rows.append((L, 784 / L, mse, 10 * np.log10(1 / mse)))
        print("L %4d done (%.0fs)" % (L, time.time() - t0), flush=True)
    print("\n     L    ratio   test MSE    PSNR")
    for L, ratio, mse, psnr in rows:
        print("  %4d  %6.1fx   %.5f   %5.1f dB" % (L, ratio, mse, psnr))


if __name__ == "__main__":
    main()
