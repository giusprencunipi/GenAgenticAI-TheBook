#!/usr/bin/env python3
"""Compute the codes plotted in the figure "The same latent space under three
models" of Chapter 4 (fig:vae_failure_modes).

    python3 supportFiles-repo/make_vae_failure_modes.py

Three models with the same 784-512-256 encoder body, a mirrored decoder, a
Bernoulli output and a two-dimensional latent space are trained on the MNIST
digits 0, 1, 3 and 7:

    (a) an autoencoder: the encoder returns one code per image;
    (b) an encoder that returns a mean and a variance, trained on
        reconstruction alone (no KL term);
    (c) a VAE: the same encoder with the KL term added.

For each model the script encodes the same 60 training images per digit and
prints their codes (the means, for (b) and (c)) as pgfplots coordinate lists,
ready to paste into vae.tex and it/vae.tex, together with the median standard
deviation the encoders of (b) and (c) assign. Nothing is drawn here: the figure
is typeset by pgfplots so that it uses the book's fonts and inks.

Requires torch and numpy; MNIST is downloaded once and cached, as in
make_mnist_figures.py. Seeded, so it reproduces on the same machine.
"""
import os, sys
import numpy as np, torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_mnist_figures import mnist           # same loader, same cache

DIGITS, PER_DIGIT, L, EPOCHS, BATCH, SEED = (0, 1, 3, 7), 60, 2, 15, 128, 0


class Model(nn.Module):
    def __init__(self, variance):
        super().__init__()
        self.variance = variance
        self.body = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                  nn.Linear(512, 256), nn.ReLU())
        self.mu, self.logvar = nn.Linear(256, L), nn.Linear(256, L)
        self.dec = nn.Sequential(nn.Linear(L, 256), nn.ReLU(),
                                 nn.Linear(256, 512), nn.ReLU(),
                                 nn.Linear(512, 784))

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h)
        if not self.variance:
            return self.dec(mu), mu, None
        logvar = self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar


def train(X, variance, kl):
    torch.manual_seed(SEED)
    model = Model(variance)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            xb = X[perm[i:i + BATCH]]
            logits, mu, logvar = model(xb)
            loss = bce(logits, xb).sum(1).mean()
            if kl:
                loss = loss + (-0.5 * (1 + logvar - mu.pow(2)
                                       - logvar.exp()).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    images, labels = mnist()
    keep = np.isin(labels, DIGITS)
    X = torch.tensor(images[keep] / 255.0, dtype=torch.float32)
    y = labels[keep]
    rng = np.random.default_rng(SEED)
    pick = {d: rng.choice(np.flatnonzero(y == d), PER_DIGIT, replace=False)
            for d in DIGITS}
    for name, variance, kl in (("a", False, False), ("b", True, False),
                               ("c", True, True)):
        model = train(X, variance, kl)
        print("%% panel (%s)" % name)
        with torch.no_grad():
            for d in DIGITS:
                h = model.body(X[pick[d]])
                mu = model.mu(h).numpy()
                print("%% digit %d" % d)
                print("coordinates {" + " ".join("(%.2f,%.2f)" % tuple(p)
                                                for p in mu) + "};")
            if variance:
                sd = torch.exp(0.5 * model.logvar(model.body(X))).numpy()
                print("%% median sd: %.3f %.3f" % tuple(np.median(sd, 0)))


if __name__ == "__main__":
    main()
