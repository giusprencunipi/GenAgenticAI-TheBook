#!/usr/bin/env python3
"""Reproduce the table of test accuracies in Chapter 3, Section "Pre-training
for Supervised Learning", in the worked example titled

    Concrete example: MNIST digit classification with few labels

The example trains an autoencoder on all 60,000 MNIST training images without
labels, and then asks how much that helps a classifier that has only a few of
the labels: 100, 1,000 or 10,000 (ten, a hundred or a thousand per digit
class). Each row of the book's table is one of those label budgets; each
column is one of four ways of using them:

    linear head on pixels    a linear softmax head trained on the 784 raw
                             pixels, with no autoencoder at all
    A: frozen encoder        the same kind of head on the codes of the frozen,
                             pre-trained encoder ("Stage 3A" in the text)
    B: fine-tuned encoder    encoder and head fine-tuned together, the encoder
                             with a learning rate ten times smaller than the
                             head's ("Stage 3B")
    B from random weights    the same network as B, trained in the same way on
                             the same labels, but starting from random weights
                             instead of the pre-trained encoder: B minus this
                             column is what the pre-training itself bought

This script runs all of them and prints the table, so that the numbers in the
book are measured rather than asserted. If the book and this script disagree,
the script is right.

    python3 supportFiles-repo/make_ae_pretraining_numbers.py

What is done, stage by stage, in the order the example gives:

  Stage 1  The autoencoder of the chapter, 784-512-256-L-256-512-784 with ReLU
           hidden layers, a linear code layer and a sigmoid output, trained on
           all 60,000 training images with the MSE loss and no labels. The
           code is widened to L = 64 exactly as the example says (the running
           example of the chapter uses L = 20).
  Stage 2  A single linear layer followed by softmax, 64 -> 10, that is
           10 x 64 + 10 = 650 parameters.
  Stage 3  The head is trained on the 100 labelled images with cross-entropy,
           either on frozen codes (A) or jointly with the encoder (B, encoder
           learning rate 1e-4, head 1e-3, as in the text).

Accuracy is measured on the 10,000 MNIST test images, which none of the stages
ever sees. With few labels the result depends noticeably on *which* images
were labelled, so Stages 2 and 3 are repeated for several independent draws
of the labels (more draws where the spread is larger) and the script reports
the mean and the standard deviation; the autoencoder of Stage 1 is trained
once and shared, since it never sees a label.

There is deliberately no early stopping and no hyperparameter search: with
100 labels there is no labelled validation set to tune on that would not be
taken out of the 100 themselves, and tuning on the test set would make the
numbers meaningless. Every configuration is trained for the same fixed number
of full-batch steps, at every label budget.

Requires torch and numpy. Seeded, so a second run reproduces the first. Runs
on the CPU in about eight minutes, most of it the 10,000-label row.
"""
import gzip, os, subprocess, sys, time, urllib.request
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".mnist-cache")          # shared with the others
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"

L = 64                       # code width used by the example
AE_EPOCHS, AE_BATCH, AE_LR = 30, 128, 1e-3
HEAD_STEPS, HEAD_LR, ENC_LR = 1000, 1e-3, 1e-4
BUDGETS = ((10, 10), (100, 3), (1000, 2))   # (labels per class, draws)
SEED = 0


def _download(url, path):
    """Fetch `url` to `path`, falling back to curl where urllib has no CA."""
    try:
        urllib.request.urlretrieve(url, path)
        return
    except Exception as exc:                       # noqa: BLE001 -- any failure
        print("%% urllib failed (%s); trying curl" % exc, file=sys.stderr)
    if subprocess.call(["curl", "-sSL", "--fail", "-o", path, url]) != 0:
        raise SystemExit("could not download " + url)


def mnist(split):
    """Download (once) and decode MNIST images in [0,1] and labels."""
    os.makedirs(CACHE, exist_ok=True)
    stem = "train" if split == "train" else "t10k"
    out = []
    for name, header in (("%s-images-idx3-ubyte.gz" % stem, 16),
                         ("%s-labels-idx1-ubyte.gz" % stem, 8)):
        path = os.path.join(CACHE, name)
        if not os.path.exists(path):
            print("downloading", name, flush=True)
            _download(MIRROR + name, path)
        with gzip.open(path, "rb") as f:
            f.read(header)
            out.append(np.frombuffer(f.read(), dtype=np.uint8))
    X = torch.tensor(out[0].reshape(-1, 784), dtype=torch.float32) / 255.0
    y = torch.tensor(out[1], dtype=torch.long)
    return X, y


class Autoencoder(nn.Module):
    """The chapter's autoencoder, with the code widened to L."""

    def __init__(self, d=784, h1=512, h2=256, z=L):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, h1), nn.ReLU(),
                                     nn.Linear(h1, h2), nn.ReLU(),
                                     nn.Linear(h2, z))          # no activation
        self.decoder = nn.Sequential(nn.Linear(z, h2), nn.ReLU(),
                                     nn.Linear(h2, h1), nn.ReLU(),
                                     nn.Linear(h1, d), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))


def stage1_pretrain(X):
    """Stage 1: train the autoencoder on all training images, no labels."""
    torch.manual_seed(SEED)
    ae = Autoencoder()
    opt = torch.optim.Adam(ae.parameters(), lr=AE_LR)
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(AE_EPOCHS):
        t0, total = time.time(), 0.0
        for idx in torch.randperm(len(X), generator=g).split(AE_BATCH):
            x = X[idx]
            loss = F.mse_loss(ae(x), x)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print("stage 1  epoch %2d  mse %.5f  (%.1fs)"
              % (epoch + 1, total / len(X), time.time() - t0), flush=True)
    return ae


def draw_labels(y, draw, per_class):
    """Pick per_class training images of each digit, independently per draw."""
    g = np.random.default_rng(1000 + draw)
    idx = [g.choice(np.flatnonzero(y.numpy() == c), per_class, replace=False)
           for c in range(10)]
    return torch.tensor(np.concatenate(idx))


def train_head(features, y, encoder=None, draw=0):
    """Stages 2 and 3: a linear softmax head, full batch, fixed step count.

    `features` is either the raw pixels or the input images; when `encoder` is
    given it is fine-tuned jointly (Strategy B) with its own learning rate.
    Returns a function mapping inputs to predicted classes.
    """
    torch.manual_seed(SEED + draw)
    in_dim = features.shape[1] if encoder is None else L
    head = nn.Linear(in_dim, 10)                   # 650 parameters when L = 64
    groups = [{"params": head.parameters(), "lr": HEAD_LR}]
    if encoder is not None:
        groups.append({"params": encoder.parameters(), "lr": ENC_LR})
    opt = torch.optim.Adam(groups)
    for _ in range(HEAD_STEPS):
        z = features if encoder is None else encoder(features)
        loss = F.cross_entropy(head(z), y)
        opt.zero_grad(); loss.backward(); opt.step()
    if encoder is None:
        return lambda x: head(x).argmax(1)
    return lambda x: head(encoder(x)).argmax(1)


def accuracy(predict, X, y):
    with torch.no_grad():
        return (predict(X) == y).float().mean().item() * 100


def main():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    Xtr, ytr = mnist("train")
    Xte, yte = mnist("test")
    print("train %d images, test %d images" % (len(Xtr), len(Xte)), flush=True)

    ae = stage1_pretrain(Xtr)
    pretrained = {k: v.clone() for k, v in ae.encoder.state_dict().items()}

    cols = ("pixels", "A frozen", "B fine-tuned", "B random init")
    table = []
    for per_class, draws in BUDGETS:
        res = {c: [] for c in cols}
        for draw in range(draws):
            idx = draw_labels(ytr, draw, per_class)
            Xl, yl = Xtr[idx], ytr[idx]

            # Baseline: the same kind of head, on the 784 raw pixels.
            res["pixels"].append(
                accuracy(train_head(Xl, yl, draw=draw), Xte, yte))

            # Strategy A: frozen encoder. The codes are computed once, with
            # no gradient, so nothing can reach the encoder's weights.
            ae.encoder.load_state_dict(pretrained)
            with torch.no_grad():
                Zl = ae.encoder(Xl)
            head_a = train_head(Zl, yl, draw=draw)
            res["A frozen"].append(
                accuracy(lambda x: head_a(ae.encoder(x)), Xte, yte))

            # Strategy B: fine-tune a fresh copy of the pre-trained encoder.
            enc_b = Autoencoder().encoder
            enc_b.load_state_dict(pretrained)
            res["B fine-tuned"].append(
                accuracy(train_head(Xl, yl, encoder=enc_b, draw=draw), Xte, yte))

            # Scratch: the same network and learning rates as B, but random
            # initial weights, so that B minus this is the pre-training's
            # own contribution.
            torch.manual_seed(500 + draw)
            enc_s = Autoencoder().encoder
            res["B random init"].append(
                accuracy(train_head(Xl, yl, encoder=enc_s, draw=draw), Xte, yte))

            print("labels %5d  draw %d  " % (10 * per_class, draw)
                  + "  ".join("%s %5.1f%%" % (c, res[c][-1]) for c in cols),
                  flush=True)
        table.append((10 * per_class, draws, res))

    print("\nTest accuracy (%), mean +- std over the draws of the labels:\n")
    print("  labels  draws  " + "".join("%16s" % c for c in cols))
    for n, draws, res in table:
        print("  %6d  %5d  " % (n, draws)
              + "".join("%10.1f +-%4.1f" % (np.mean(res[c]), np.std(res[c]))
                        for c in cols))
    print("\nPre-training's own contribution, B fine-tuned minus B random init:")
    for n, draws, res in table:
        gain = np.array(res["B fine-tuned"]) - np.array(res["B random init"])
        print("  %6d labels  %+5.1f points" % (n, gain.mean()))


if __name__ == "__main__":
    main()
