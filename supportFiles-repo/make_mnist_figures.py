#!/usr/bin/env python3
"""Regenerate the three MNIST images used by Chapter 4, in the figures captioned
"The latent space of a real VAE, decoded" (fig:mnist_manifold) and "Two
straight lines through the latent space" (fig:mnist_interp).

    mnist_manifold.png      a 19x19 grid of decoded latent codes
    mnist_interp_38.png     eleven steps from a 3 to an 8
    mnist_interp_17.png     eleven steps from a 1 to a 7

These are the only figures in the chapter that cannot be drawn in TikZ: they
are the output of a network that has actually been trained, so they have to be
computed. This script does the whole thing from nothing -- it downloads MNIST,
trains the model, and writes the images -- and takes about three minutes.

    python3 supportFiles-repo/make_mnist_figures.py

The model is deliberately the one the chapter describes: a 784-512-256 encoder
with a mean head and a log-variance head, a mirrored decoder with a Bernoulli
(sigmoid) output, and the loss of the section "The VAE Loss Function"
(eq:vae_loss), with the reconstruction term
summed over pixels and averaged over the batch exactly as the remark
"Reduction options for BCE" in the ELBO section (sec:recon_term) says.
The one departure is the latent width: L = 2 rather than 20, because the point
of the figures is to put the whole latent space on a page.

The images are written as 8-bit greyscale, dark ink on white. Greyscale is
achromatic and so falls outside the book's four-colour budget; a colourmap
would break it.

Requires torch, numpy, scipy and pillow. Seeded, so it reproduces.
"""
import gzip, os, subprocess, sys, time, urllib.request
import numpy as np, torch, torch.nn as nn
from scipy.stats import norm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".mnist-cache")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
L, EPOCHS, BATCH, SEED = 2, 60, 128, 0


def _download(url, path):
    """Fetch `url` to `path`.

    urllib is tried first. On machines whose Python has no usable CA bundle it
    raises URLError, which is common enough on macOS to be worth handling, so
    curl is tried next; it is present wherever this script is likely to run and
    uses the system trust store.
    """
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
    return out[0].reshape(-1, 784), out[1]


class VAE(nn.Module):
    def __init__(self, x_dim=784, h1=512, h2=256, z_dim=L):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(x_dim, h1), nn.ReLU(),
                                  nn.Linear(h1, h2), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h2, z_dim), nn.Linear(h2, z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(),
                                 nn.Linear(h1, x_dim))

    def encode(self, x):
        h = self.body(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.dec(z), mu, logvar


def train(X, dev):
    model, N = VAE().to(dev), X.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(EPOCHS):
        t0, perm, tot_r, tot_k = time.time(), torch.randperm(N, device=dev), 0.0, 0.0
        for i in range(0, N, BATCH):
            xb = X[perm[i:i + BATCH]]
            logits, mu, logvar = model(xb)
            rec = bce(logits, xb).sum(1).mean()          # summed over pixels,
            kl = (-0.5 * (1 + logvar - mu.pow(2)         # averaged over the batch
                          - logvar.exp()).sum(1)).mean()
            opt.zero_grad(); (rec + kl).backward(); opt.step()
            tot_r += rec.item() * xb.shape[0]; tot_k += kl.item() * xb.shape[0]
        print("epoch %2d  recon %7.2f  kl %6.2f  (%.1fs)"
              % (ep + 1, tot_r / N, tot_k / N, time.time() - t0), flush=True)
    return model.eval()


def to_png(a, up=4):
    """float in [0,1] -> dark-ink-on-white uint8, upscaled without smoothing."""
    a = (255 * (1.0 - np.clip(a, 0, 1))).astype(np.uint8)
    return Image.fromarray(np.kron(a, np.ones((up, up), np.uint8)))


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, flush=True)

    raw, labels = mnist()
    x = raw.astype(np.float32) / 255.0
    model = train(torch.from_numpy(x).to(dev), dev)

    def decode(z):
        with torch.no_grad():
            return torch.sigmoid(
                model.dec(torch.as_tensor(z, dtype=torch.float32).to(dev))
            ).cpu().numpy()

    # ── fig:mnist_manifold: a regular grid over the prior, decoded ─────────
    # Quantiles, not a uniform box: a uniform grid over a wide box spends most
    # of its cells where the prior puts no weight, and the edges come out
    # empty.
    n = 19
    q = norm.ppf(np.linspace(0.008, 0.992, n))
    grid = np.zeros((n * 28, n * 28), np.float32)
    for i, z2 in enumerate(q[::-1]):                     # top row = largest z2
        for j, z1 in enumerate(q):
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = decode([[z1, z2]]).reshape(28, 28)
    to_png(grid).save(os.path.join(HERE, "mnist_manifold.png"))
    print("wrote mnist_manifold.png  (z from %.2f to %.2f)" % (q[0], q[-1]))

    # ── fig:mnist_interp: two straight lines between encoded digits ────────
    with torch.no_grad():
        mus = model.encode(torch.from_numpy(x[:20000]).to(dev))[0].cpu().numpy()
    y = labels[:20000]

    def typical(c):
        """The example of class c closest to that class's mean code, so that
        the endpoints are ordinary digits rather than outliers."""
        idx = np.where(y == c)[0]
        return idx[np.argmin(((mus[idx] - mus[idx].mean(0)) ** 2).sum(1))]

    for a, b in ((3, 8), (1, 7)):
        za, zb = mus[typical(a)], mus[typical(b)]
        zs = np.array([(1 - t) * za + t * zb for t in np.linspace(0, 1, 11)])
        strip = np.concatenate(list(decode(zs).reshape(-1, 28, 28)), axis=1)
        to_png(strip).save(os.path.join(HERE, "mnist_interp_%d%d.png" % (a, b)))
        print("wrote mnist_interp_%d%d.png  z %s -> %s"
              % (a, b, np.round(za, 2), np.round(zb, 2)))


if __name__ == "__main__":
    main()
