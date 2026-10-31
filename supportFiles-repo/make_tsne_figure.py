#!/usr/bin/env python3
"""Regenerate the t-SNE projection used by the appendix on word embeddings.

    tsne_seeds.png      the same 32 words projected twice, with two random
                        seeds, so that what survives the projection and what
                        does not can be seen side by side

    python3 supportFiles-repo/make_tsne_figure.py      (run from the repo root)
    python3 supportFiles-repo/make_tsne_figure.py --lang it --out it/tsne_seeds.png

The second form writes the Italian edition's copy: same runs, only the legend
translated. The word labels are the data (GloVe is English) and "seed" is
used untranslated in the Italian text, so nothing else changes. The Italian
edition keeps the file under it/ with the same name as the English one, which
LaTeX finds first; it/README.md explains.

One caveat, and the --panels-from option that answers it. t-SNE is a gradient
descent on a non-convex objective, so the layout it reaches depends on the
last bit of every floating-point operation, and the same seed gives a
different (equally valid) picture on a different machine or BLAS. The caption
in the appendix describes the committed picture (where royalty lands, which
animals sit apart), so a relabelled edition must keep those panels exactly.
    --panels-from tsne_seeds.png
takes the two panels from an existing figure and replaces only the legend
strip below them with the newly drawn one, so the Italian file differs from
the English one in the legend and nowhere else. Without it, a fresh run on
another machine would redraw the panels and silently falsify the caption.

The appendix argues that projections of embeddings are over-read: that
distances inside a visible cluster carry information and distances between
clusters carry much less, and that the layout on the page is largely an
accident of initialisation. An argument of that shape should not be made with
a diagram drawn by hand, so the figure is two real runs of t-SNE on real
vectors, differing only in the seed.

It also reports the numbers the text quotes, and it reports them over twelve
runs rather than one, because one run says nothing about reliability.
Correlating a projection's pairwise distances against the pairwise distances in
the original hundred-dimensional space gives, inside a family, 0.74 on average
and never below 0.49. Between families it gives 0.30 on average and anywhere
from 0.02 to 0.61 -- which is not a weaker signal so much as a lottery.

One thing does survive every run: countries and capitals stay the closest two
families, at 0.44 against 1.33 for countries and animals, with the two ranges
not overlapping. So the coarse fact of which families neighbour which
is real; the geometry between them is not. Nor is the position on the page:
across twelve seeds royalty comes out at the top of the picture seven times and
at the bottom twice.

t-SNE is implemented here in about fifty lines of numpy rather than imported.
It is a small algorithm -- match a distribution over pairs in the original
space with one in the plane, minimise the KL divergence between them by
gradient descent -- and a book that spends four hundred pages refusing to leave
mechanisms unopened should not import this one. It is also the only dependency
that would otherwise be needed beyond numpy and matplotlib.

The vectors are the GloVe 6B release, fetched by make_analogy_numbers.py; run
that first, or pass --cache to point at the file it wrote.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "tsne_seeds.png"
CACHE = "glove.6B.100d.txt"
DIM = 100
# Two runs that differ. Any two seeds give the same families and a different
# arrangement, but these two make it plainest: the correlation between their
# inter-cluster distances is 0.06, and royalty moves from the top of one panel
# to below the countries in the other.
SEEDS = (6, 9)
PERPLEXITY = 6.0

FAMILIES = {
    "animals":   ["cat", "dog", "lion", "wolf", "tiger", "horse", "bear", "rabbit"],
    "royalty":   ["king", "queen", "prince", "princess", "throne", "crown",
                  "monarch", "duke"],
    "countries": ["france", "germany", "italy", "spain", "japan", "canada",
                  "brazil", "poland"],
    "capitals":  ["paris", "berlin", "rome", "madrid", "tokyo", "ottawa",
                  "brasilia", "warsaw"],
}
MARKERS = {"animals": "o", "royalty": "s", "countries": "^", "capitals": "D"}
# Legend text by edition. The family names are also the keys of FAMILIES, so
# English needs no table; the Italian is the only translated element.
LEGEND = {"en": {f: f for f in FAMILIES},
          "it": {"animals": "animali", "royalty": "regalità",
                 "countries": "paesi", "capitals": "capitali"}}
# A few words per family are labelled, enough to anchor the reader; labelling
# all thirty-two is unreadable at the width of a page.
ANCHORS = {"animals": {"cat", "tiger", "horse"},
           "royalty": {"king", "queen", "throne"},
           "countries": {"france", "japan", "brazil"},
           "capitals": {"paris", "tokyo", "ottawa"}}


def joint_p(X, perplexity, tol=1e-5):
    """P_ij: how likely j is to be picked as a neighbour of i, symmetrised.

    Each point gets its own Gaussian width, found by bisection so that the
    entropy of its neighbour distribution matches the requested perplexity.
    That is what makes t-SNE adaptive: a point in a dense region gets a narrow
    kernel, one in a sparse region a wide one.
    """
    n = len(X)
    sq = (X ** 2).sum(1)
    D = np.add.outer(sq, sq) - 2 * X @ X.T
    np.fill_diagonal(D, 0.0)
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        lo, hi, beta = -np.inf, np.inf, 1.0
        Di = np.delete(D[i], i)
        for _ in range(60):
            Pi = np.exp(-Di * beta)
            s = Pi.sum() or 1e-12
            entropy = np.log(s) + beta * (Di * Pi).sum() / s
            if abs(entropy - target) < tol:
                break
            if entropy > target:                      # too flat: sharpen
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, np.arange(n) != i] = Pi / s
    return np.maximum((P + P.T) / (2 * n), 1e-12)


def tsne(X, seed, perplexity=PERPLEXITY, iters=1000, lr=200.0, exaggeration=12.0):
    """Minimise KL(P || Q) over positions in the plane, Q being a Student-t.

    The heavy tail of the Student-t is the whole trick: it lets points that are
    far apart in the original space be pushed very far apart on the page
    without paying much, which is what stops everything collapsing into one
    blob. It is also why distances between clusters cannot be read literally.
    """
    rng = np.random.default_rng(seed)
    P = joint_p(X, perplexity)
    Y = rng.normal(0, 1e-4, (len(X), 2))
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for step in range(iters):
        early = step < 250
        sq = (Y ** 2).sum(1)
        num = 1.0 / (1.0 + np.add.outer(sq, sq) - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / num.sum(), 1e-12)
        W = ((P * exaggeration if early else P) - Q) * num
        grad = 4 * ((np.diag(W.sum(1)) - W) @ Y)
        gains = np.maximum(np.where(np.sign(grad) != np.sign(velocity),
                                    gains + 0.2, gains * 0.8), 0.01)
        velocity = (0.5 if early else 0.8) * velocity - lr * gains * grad
        Y = Y + velocity
        Y -= Y.mean(0)
    return Y


def load(path, wanted):
    keep = {w for ws in wanted.values() for w in ws}
    found = {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ")
        if len(parts) == DIM + 1 and parts[0] in keep:
            found[parts[0]] = np.array(parts[1:], dtype=np.float64)
    missing = keep - set(found)
    if missing:
        raise SystemExit(f"not in the vectors: {sorted(missing)}")
    words = [w for ws in wanted.values() for w in ws]
    labels = [f for f, ws in wanted.items() for _ in ws]
    return words, labels, np.array([found[w] for w in words])


def scaled_distances(Y):
    """Pairwise distances, divided by their own mean so that two layouts of
    different overall size can be compared."""
    sq = (Y ** 2).sum(1)
    D = np.sqrt(np.maximum(np.add.outer(sq, sq) - 2 * Y @ Y.T, 0))
    return D / D[np.triu_indices(len(Y), 1)].mean()


def survey(X, labels, seeds):
    """Run the projection many times and report what is stable across runs.

    One run tells you nothing about reliability. The quantities the appendix
    quotes are these, over `seeds` independent projections of the same
    vectors: how well distances inside a family are preserved, how well
    distances between families are, and whether the two families that are
    genuinely closest stay closest.
    """
    labels = np.array(labels)
    iu = np.triu_indices(len(X), 1)
    same = labels[iu[0]] == labels[iu[1]]
    O = scaled_distances(X)

    def between_families(D, a, b):
        m = (((labels[iu[0]] == a) & (labels[iu[1]] == b)) |
             ((labels[iu[0]] == b) & (labels[iu[1]] == a)))
        return D[iu][m].mean()

    within, across, near, far = [], [], [], []
    for seed in seeds:
        D = scaled_distances(tsne(X, seed=seed))
        within.append(np.corrcoef(O[iu][same], D[iu][same])[0, 1])
        across.append(np.corrcoef(O[iu][~same], D[iu][~same])[0, 1])
        near.append(between_families(D, "countries", "capitals"))
        far.append(between_families(D, "countries", "animals"))

    def line(name, xs):
        print(f"  {name:<34} mean {np.mean(xs):.2f}   "
              f"range {min(xs):.2f} to {max(xs):.2f}")

    print(f"\nover {len(seeds)} independent runs:")
    line("distances kept within a family", within)
    line("distances kept between families", across)
    line("distance countries to capitals", near)
    line("distance countries to animals", far)

    tops = {}
    for seed in seeds:
        Y = tsne(X, seed=seed)
        centres = {f: Y[labels == f].mean(0) for f in FAMILIES}
        top = max(centres, key=lambda f: centres[f][1])
        bottom = min(centres, key=lambda f: centres[f][1])
        tops.setdefault(top, []).append(seed)
        tops.setdefault("bottom:" + bottom, []).append(seed)
    print("\n  which family lands at the top, by seed:")
    for k in sorted(k for k in tops if not k.startswith("bottom:")):
        print(f"    {k:<12} {len(tops[k])} of {len(list(seeds))}")
    print("\n  which family lands at the bottom, by seed:")
    for k in sorted(k for k in tops if k.startswith("bottom:")):
        print(f"    {k[7:]:<12} {len(tops[k])} of {len(list(seeds))}")


def relabel(source, target):
    """Copy the panels of `source` over `target`, keeping `target`'s legend.

    The legend is the lowest band of non-white rows; the cut is made in the
    white gap above it, and both files must be white there, which they are
    when both come from draw() at the same size and dpi."""
    from PIL import Image
    import numpy as np
    src, tgt = Image.open(source).convert("RGB"), Image.open(target).convert("RGB")
    if src.size != tgt.size:
        raise SystemExit(f"{source} is {src.size}, {target} is {tgt.size}: "
                         "regenerate both at the same size and dpi")
    inked = (np.asarray(tgt.convert("L")) < 250).any(axis=1)
    legend_top = max(y for y in range(len(inked) - 1)
                     if not inked[y] and inked[y + 1:].any() and inked[y + 1])
    cut = legend_top - 40
    for name, im in ((source, src), (target, tgt)):
        band = np.asarray(im.convert("L"))[cut - 20:cut + 20]
        if (band < 250).any():
            raise SystemExit(f"{name} is not white around row {cut}; "
                             "the layout changed, do not splice")
    out = tgt.copy()
    out.paste(src.crop((0, 0, src.width, cut)), (0, 0))
    out.save(target, dpi=tgt.info.get("dpi", (400, 400)))


# Where a label may sit relative to its marker, in points, with the text
# alignment that keeps it on that side. The first is the default; the others
# are tried only when it would cover another label or marker.
LABEL_SPOTS = [((3.5, 2.5), "left", "bottom"), ((3.5, -2.5), "left", "top"),
               ((-3.5, 2.5), "right", "bottom"), ((-3.5, -2.5), "right", "top"),
               ((0, 4.5), "center", "bottom"), ((0, -4.5), "center", "top"),
               ((5.5, 0), "left", "center"), ((-5.5, 0), "right", "center")]


def place_labels(fig, ax, Y, items):
    """Label the anchor words without covering one another or the markers.

    In a tight cluster the default spot, up and to the right, lands on the
    neighbouring marker or label. Each label takes the first spot in
    LABEL_SPOTS that overlaps nothing already on the panel and stays inside
    it, or the least-overlapping spot if none is free."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    half = 0.75 * np.sqrt(20) * fig.dpi / 72      # marker half-width, with a margin
    pts = ax.transData.transform(Y)
    taken = [Bbox.from_extents(x - half, y - half, x + half, y + half)
             for x, y in pts]
    frame = ax.get_window_extent(r)
    def overlap(a, b):
        w = min(a.x1, b.x1) - max(a.x0, b.x0)
        h = min(a.y1, b.y1) - max(a.y0, b.y0)
        return max(w, 0) * max(h, 0)
    for i, w in items:
        best = None
        for (dx, dy), ha, va in LABEL_SPOTS:
            t = ax.annotate(w, (Y[i, 0], Y[i, 1]), textcoords="offset points",
                            xytext=(dx, dy), ha=ha, va=va, fontsize=5.6,
                            color="0.35", zorder=4)
            bb = t.get_window_extent(r).expanded(1.04, 1.1)
            cost = sum(overlap(bb, o) for o in taken)
            cost += bb.width * bb.height - overlap(bb, frame)
            if best is None or cost < best[0]:
                if best is not None:
                    best[1].remove()
                best = (cost, t, bb)
            else:
                t.remove()
            if cost == 0:
                break
        taken.append(best[2])


def draw(path, layouts, words, labels, lang="en"):
    fig, axes = plt.subplots(1, 2, figsize=(5.05, 3.3))
    fitted = []
    for ax, (seed, Y) in zip(axes, layouts.items()):
        # Fit each panel to its own data. A shared fixed range wastes most of
        # the picture on margin, and comparing absolute positions between the
        # panels is precisely what the figure exists to discourage.
        Y = Y - (Y.max(0) + Y.min(0)) / 2
        Y = Y / np.abs(Y).max()
        fitted.append(Y)
        for fam in FAMILIES:
            m = [i for i, f in enumerate(labels) if f == fam]
            ax.scatter(Y[m, 0], Y[m, 1], marker=MARKERS[fam], s=20,
                       facecolors="white", edgecolors="0.2",
                       linewidths=0.7, zorder=3)
        ax.set_title(f"seed {seed}", fontsize=7, color="0.25", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.18, 1.18)
        ax.set_ylim(-1.18, 1.18)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_color("0.75")
            s.set_linewidth(0.5)
    handles = [plt.Line2D([], [], marker=MARKERS[f], linestyle="",
                          markerfacecolor="white", markeredgecolor="0.2",
                          markeredgewidth=0.7, markersize=3.8,
                          label=LEGEND[lang][f])
               for f in FAMILIES]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.2, handletextpad=0.25, columnspacing=1.4,
               bbox_to_anchor=(0.5, -0.015))
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    # Labels go in last, once the panels have their final size, because
    # whether two of them collide depends on it.
    anchored = [(i, w) for i, w in enumerate(words) if w in ANCHORS[labels[i]]]
    for ax, Y in zip(axes, fitted):
        place_labels(fig, ax, Y, anchored)
    plt.savefig(path, dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=CACHE,
                    help="the GloVe file make_analogy_numbers.py writes")
    ap.add_argument("--runs", type=int, default=12,
                    help="how many seeds to survey for the reported spread")
    ap.add_argument("--lang", choices=sorted(LEGEND), default="en",
                    help="language of the legend (the words are GloVe's, English)")
    ap.add_argument("--out", default=OUT,
                    help="where to write the figure (it/tsne_seeds.png for the Italian edition)")
    ap.add_argument("--panels-from", metavar="PNG",
                    help="keep the panels of this existing figure and replace only "
                         "its legend strip with the newly drawn one (see the docstring)")
    args = ap.parse_args()

    words, labels, X = load(args.cache, FAMILIES)
    print(f"{len(words)} words in {len(FAMILIES)} families, {X.shape[1]} dimensions")
    layouts = {s: tsne(X, seed=s) for s in SEEDS}
    draw(args.out, layouts, words, labels, args.lang)
    if args.panels_from:
        relabel(args.panels_from, args.out)
    print(f"  wrote {args.out}")

    # How differently the two drawn panels arrange the families.
    import itertools
    lab = np.array(labels)
    def centre_gaps(Y):
        c = {f: Y[lab == f].mean(0) for f in FAMILIES}
        return np.array([np.linalg.norm(c[a] - c[b])
                         for a, b in itertools.combinations(FAMILIES, 2)])
    a, b = (centre_gaps(Y) for Y in layouts.values())
    print(f"  the two panels' inter-cluster distances correlate at "
          f"{np.corrcoef(a, b)[0, 1]:.2f}")
    survey(X, labels, range(1, args.runs + 1))


if __name__ == "__main__":
    main()
