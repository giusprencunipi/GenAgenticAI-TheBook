#!/usr/bin/env python3
"""Regenerate the figures of Chapter 5 that need a model to be trained.

The figure "Mode collapse as it happens" (fig:mode_hopping), on a ring of eight
Gaussians, is the output of a GAN
that was actually trained, so it has to be computed. This script trains it and
prints the sample coordinates as the TikZ body of the figure, ready to be
pasted into gan.tex, together with the mode counts quoted in the caption.

The figure comparing Jensen-Shannon and Wasserstein (fig:js_vs_wasserstein)
needs no training: the
Wasserstein distance between two equal-width Gaussians a distance theta apart
is exactly theta, and the JS divergence is a one-dimensional integral. The
script evaluates it and prints the coordinate list.

The figure of MNIST samples from a GAN and from a VAE (fig:gan_vs_vae_samples)
is two trained models as
well. Both are trained here, matched where the two methods allow, and written
as the two PNGs the chapter includes.

    python3 supportFiles-repo/make_gan_figures.py            # all three
    python3 supportFiles-repo/make_gan_figures.py mnist      # fig:gan_vs_vae_samples only
    python3 supportFiles-repo/make_gan_figures.py ring       # fig:mode_hopping only
    python3 supportFiles-repo/make_gan_figures.py js         # fig:js_vs_wasserstein only

The ring model is a deliberately ordinary GAN -- two small leaky-ReLU stacks,
the non-saturating generator loss of the section "The Minimax Objective: One
Game, Two Players", Adam, two discriminator
steps per generator step. Nothing is done to induce the collapse; it is what
this configuration does. The seed is fixed, so the figure reproduces.

Requires torch and numpy.
"""
import gzip
import os
import subprocess
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".mnist-cache")
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

K, RADIUS, SPREAD, SEED = 8, 2.0, 0.06, 13
SNAPSHOTS = (500, 2000, 4000, 9000)
CENTRES = np.stack([[RADIUS * np.cos(2 * np.pi * k / K),
                     RADIUS * np.sin(2 * np.pi * k / K)] for k in range(K)])


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


def modes_covered(samples, tol=0.35, floor=0.02):
    """How many of the eight modes hold at least `floor` of the samples."""
    d = ((samples[:, None, :] - CENTRES[None, :, :]) ** 2).sum(-1) ** 0.5
    near = d.min(1) < tol
    counts = np.bincount(d.argmin(1)[near], minlength=K)
    return int((counts > floor * len(samples)).sum()), float(near.mean()), counts


def ring():
    import torch, torch.nn as nn
    torch.manual_seed(SEED); np.random.seed(SEED)

    def real(n):
        i = np.random.randint(0, K, n)
        return torch.tensor(CENTRES[i] + SPREAD * np.random.randn(n, 2),
                            dtype=torch.float32)

    def mlp(i, o, h=128, layers=2):
        m = [nn.Linear(i, h), nn.LeakyReLU(0.2)]
        for _ in range(layers - 1):
            m += [nn.Linear(h, h), nn.LeakyReLU(0.2)]
        return nn.Sequential(*m, nn.Linear(h, o))

    G, D = mlp(2, 2), mlp(2, 1)
    oG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=3e-3, betas=(0.5, 0.9))
    bce, B, snaps = nn.BCEWithLogitsLoss(), 256, {}

    for step in range(1, max(SNAPSHOTS) + 1):
        for _ in range(2):                                  # discriminator
            x, z = real(B), torch.randn(B, 2)
            with torch.no_grad():
                fake = G(z)
            loss = bce(D(x), torch.ones(B, 1)) + bce(D(fake), torch.zeros(B, 1))
            oD.zero_grad(); loss.backward(); oD.step()
        z = torch.randn(B, 2)                               # generator
        lossG = bce(D(G(z)), torch.ones(B, 1))              # non-saturating
        oG.zero_grad(); lossG.backward(); oG.step()
        if step in SNAPSHOTS:
            with torch.no_grad():
                snaps[step] = G(torch.randn(800, 2)).numpy()
            n, frac, counts = modes_covered(snaps[step])
            print("%% step %5d  modes %d/8  on-mode %.0f%%  counts %s"
                  % (step, n, 100 * frac, list(counts)), file=sys.stderr)

    rng = np.random.default_rng(0)

    def coords(a, n=230):
        idx = rng.choice(len(a), size=min(n, len(a)), replace=False)
        # slash-separated pairs: \foreach splits its list on commas, so a
        # coordinate written (x,y) would be torn in half
        return ", ".join("%.3f/%.3f" % (x, y) for x, y in a[idx])

    panels = [("real data", real(800).numpy(), None)]
    labels = {500: "after 500 steps", 2000: "2\\,000 steps",
              4000: "4\\,000 steps", 9000: "9\\,000 steps"}
    for st in SNAPSHOTS:
        panels.append((labels[st], snaps[st], modes_covered(snaps[st])[0]))

    print("\\begin{figure}[ht]\n\\centering\n\\resizebox{\\textwidth}{!}{%"
          "\n\\begin{tikzpicture}[font=\\small]")
    for i, (title, arr, n) in enumerate(panels):
        ink = "inkrule" if n is None else "inkfour"
        print("  \\begin{scope}[xshift=%.2fcm]" % (i * 6.3))
        print("    \\draw[inkrule!45] (-2.9,-2.9) rectangle (2.9,2.9);")
        print("    \\foreach \\k in {0,...,7}")
        print("      \\draw[inkrule!45, thin] ({2*cos(\\k*45)},{2*sin(\\k*45)})"
              " circle (0.30);")
        print("    \\begin{scope}")
        print("      \\clip (-2.9,-2.9) rectangle (2.9,2.9);")
        print("      \\foreach \\a/\\b in {%s}" % coords(arr))
        print("        \\fill[%s] (\\a,\\b) circle (0.05);" % ink)
        print("    \\end{scope}")
        print("    \\node[font=\\large\\bfseries, text=inkrule] at (0,3.45) {%s};"
              % title)
        if n is None:
            print("    \\node[font=\\large, text=inkrule] at (0,-3.45)"
                  " {eight modes};")
        else:
            col = "inkrule" if n == K else "inkfour"
            print("    \\node[font=\\large, text=%s] at (0,-3.45)"
                  " {%d of 8 modes};" % (col, n))
        print("  \\end{scope}")
    print("\\end{tikzpicture}%\n}")


def js_divergence(theta, sd, grid=200001, span=12.0):
    """JS between N(0,sd^2) and N(theta,sd^2), by numerical integration."""
    x = np.linspace(-span, span + theta, grid)
    norm = lambda mu: np.exp(-(x - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
    p, q = norm(0.0), norm(theta)
    m, eps = 0.5 * (p + q), 1e-300
    kl = lambda a, b: np.trapz(np.where(a > eps, a * np.log((a + eps) / (b + eps)), 0.0), x)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mnist_pixels():
    """MNIST training images as floats in [0,1], downloaded once and cached."""
    os.makedirs(CACHE, exist_ok=True)
    name = "train-images-idx3-ubyte.gz"
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        print("%% downloading " + name, file=sys.stderr)
        _download(MIRROR + name, path)
    with gzip.open(path, "rb") as f:
        f.read(16)
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    return raw.reshape(-1, 784).astype(np.float32) / 255.0


def mnist(latent=64, epochs=40, batch=128):
    """Train a GAN and a VAE on MNIST and write the two sample grids.

    The two are matched where the methods allow: same latent width, same layer
    sizes in the generator and in the decoder, same data, same epoch count.
    They differ where the methods differ -- the GAN's generator ends in a tanh
    on data scaled to [-1,1], the VAE's decoder in a sigmoid on data in [0,1],
    each being the convention for its own objective.
    """
    import torch, torch.nn as nn
    from PIL import Image
    torch.manual_seed(0); np.random.seed(0)
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print("%% device " + dev, file=sys.stderr)

    x = mnist_pixels()
    X01 = torch.from_numpy(x).to(dev)          # VAE: Bernoulli on [0,1]
    X11 = torch.from_numpy(x * 2 - 1).to(dev)  # GAN: tanh output on [-1,1]
    N, L, B = X01.shape[0], latent, batch

    G = nn.Sequential(nn.Linear(L, 256), nn.BatchNorm1d(256), nn.ReLU(),
                      nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                      nn.Linear(512, 784), nn.Tanh()).to(dev)
    Dn = nn.Sequential(nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                       nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                       nn.Linear(256, 1)).to(dev)
    oG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    oD = torch.optim.Adam(Dn.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(B, 1, device=dev), torch.zeros(B, 1, device=dev)
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N - B, B):
            z = torch.randn(B, L, device=dev)
            with torch.no_grad():
                fake = G(z)
            loss = bce(Dn(X11[perm[i:i + B]]), ones) + bce(Dn(fake), zeros)
            oD.zero_grad(); loss.backward(); oD.step()
            z = torch.randn(B, L, device=dev)
            lg = bce(Dn(G(z)), ones)            # non-saturating
            oG.zero_grad(); lg.backward(); oG.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print("%% GAN epoch %2d" % ep, file=sys.stderr)
    G.eval()
    with torch.no_grad():
        gan_samples = ((G(torch.randn(64, L, device=dev)) + 1) / 2).cpu().numpy()

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(784, 512), nn.ReLU(),
                                      nn.Linear(512, 256), nn.ReLU())
            self.mu, self.lv = nn.Linear(256, L), nn.Linear(256, L)
            self.dec = nn.Sequential(nn.Linear(L, 256), nn.ReLU(),
                                     nn.Linear(256, 512), nn.ReLU(),
                                     nn.Linear(512, 784))

        def forward(self, x):
            h = self.body(x)
            mu, lv = self.mu(h), self.lv(h)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            return self.dec(z), mu, lv

    V = VAE().to(dev)
    oV = torch.optim.Adam(V.parameters(), lr=1e-3)
    pix = nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(epochs):
        perm = torch.randperm(N, device=dev)
        for i in range(0, N - B, B):
            xb = X01[perm[i:i + B]]
            logits, mu, lv = V(xb)
            rec = pix(logits, xb).sum(1).mean()
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
            oV.zero_grad(); (rec + kl).backward(); oV.step()
        if ep % 10 == 0 or ep == epochs - 1:
            print("%% VAE epoch %2d" % ep, file=sys.stderr)
    V.eval()
    with torch.no_grad():
        vae_samples = torch.sigmoid(V.dec(torch.randn(64, L, device=dev))).cpu().numpy()

    def grid(a, n=8, up=4):
        im = np.zeros((n * 28, n * 28), np.float32)
        for k in range(n * n):
            r, c = divmod(k, n)
            im[r*28:(r+1)*28, c*28:(c+1)*28] = a[k].reshape(28, 28)
        g = (255 * (1 - np.clip(im, 0, 1))).astype(np.uint8)
        return Image.fromarray(np.kron(g, np.ones((up, up), np.uint8)))

    grid(gan_samples).save(os.path.join(HERE, "gan_mnist_samples.png"))
    grid(vae_samples).save(os.path.join(HERE, "gan_mnist_vae_samples.png"))
    print("%% wrote gan_mnist_samples.png and gan_mnist_vae_samples.png",
          file=sys.stderr)


def js(sd=0.3):
    thetas = np.linspace(0, 3, 121)
    vals = [js_divergence(t, sd) for t in thetas]
    print("%% sigma = %.1f, log 2 = %.4f" % (sd, np.log(2)), file=sys.stderr)
    for t in (0.0, 1.0, 2.0, 3.0):
        print("%%   JS(%.1f) = %.4f" % (t, js_divergence(t, sd)), file=sys.stderr)
    print(" ".join("(%.3f,%.4f)" % (t, v) for t, v in zip(thetas, vals)))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("mnist", "all"):
        mnist()
    if what in ("ring", "all", "both"):
        ring()
    if what in ("js", "all", "both"):
        js()
