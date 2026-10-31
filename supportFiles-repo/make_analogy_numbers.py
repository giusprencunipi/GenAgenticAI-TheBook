#!/usr/bin/env python3
"""Recompute the numbers in the analogy example of the Word Embeddings appendix.

    python3 supportFiles-repo/make_analogy_numbers.py

The example prints the eight words nearest to v_king - v_man + v_woman, the
cosine of that vector with v_queen, the cosine between the two supposedly equal
offsets, and the same query for paris - france + germany. Every one of those
figures comes out of this script, so that nothing in the example rests on
vectors chosen to make it work.

The vectors are the published GloVe 6B release (Wikipedia 2014 + Gigaword 5,
400k lowercase words, 100 dimensions). They are not in this repository -- the
archive is 822 MB -- so the script fetches the part of it that it needs.

The fetch is worth a word. glove.6B.zip stores its four files deflated and in
sequence, 100d first, so the 100d vectors are a single deflate stream starting
at a known offset. A ranged GET followed by a streaming inflate therefore gives
the vectors without pulling the other three dimensionalities, which is 134 MB
instead of 822. Pass --prefix to stop after the most frequent 41,562 words:
that is 14 MB, it takes seconds, and every number below is identical, since the
words that win these queries are common ones. The default fetches the whole
vocabulary, which is what the appendix claims was searched.

No model is trained here and no vector is modified. The whole script is a
nearest-neighbour lookup, which is all the analogy claim ever was.
"""

import argparse
import itertools
import os
import random
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zlib

import numpy as np

URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
CACHE = "glove.6B.100d.txt"
PREFIX_BYTES = 14_000_000          # enough for the ~41.5k most frequent words
DIM = 100

# The queries the appendix prints, as (a, b, c, expected): a - b + c ~= expected.
QUERIES = [
    ("king", "man", "woman", "queen"),
    ("paris", "france", "germany", "berlin"),
]

# Pairs whose difference is supposed to be the same offset.
OFFSETS = [
    ("king", "man", "queen", "woman"),
    ("paris", "france", "berlin", "germany"),
]

# Families of pairs that are supposed to share one direction, for the subspace
# table. The unrelated pairs are the control: whatever they score is the noise
# floor, and the others mean nothing without it.
FAMILIES = {
    "country -> capital": [("paris", "france"), ("berlin", "germany"),
                           ("rome", "italy"), ("madrid", "spain"),
                           ("lisbon", "portugal"), ("athens", "greece"),
                           ("warsaw", "poland"), ("moscow", "russia"),
                           ("tokyo", "japan"), ("ottawa", "canada")],
    "present -> past":    [("walked", "walk"), ("talked", "talk"),
                           ("played", "play"), ("worked", "work"),
                           ("looked", "look"), ("asked", "ask"),
                           ("wanted", "want"), ("started", "start")],
    "singular -> plural": [("cars", "car"), ("dogs", "dog"),
                           ("books", "book"), ("houses", "house"),
                           ("cities", "city"), ("children", "child"),
                           ("hands", "hand"), ("years", "year")],
}


def ranged_get(url, byte_range):
    """GET a byte range, falling back to curl when Python has no CA bundle.

    A python.org build on macOS ships without certificates until its
    Install Certificates command is run, and this is exactly the kind of
    script someone runs once on a fresh machine. curl has its own trust
    store and is present on every platform this book assumes.
    """
    req = urllib.request.Request(url, headers={"Range": f"bytes={byte_range}"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.read()
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY" not in str(e.reason):
            raise
        print("  python has no CA bundle; falling back to curl", file=sys.stderr)
        out = subprocess.run(["curl", "-sSL", "-r", byte_range, url],
                             capture_output=True)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.decode()[:400]) from e
        return out.stdout


def fetch(path, prefix_only):
    """Download the 100d member of glove.6B.zip and inflate it to `path`."""
    end = PREFIX_BYTES if prefix_only else ""
    print(f"fetching {'a prefix of ' if prefix_only else ''}{URL} ...",
          file=sys.stderr)
    blob = ranged_get(URL, f"0-{end}")

    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError("not a zip local file header; the archive moved")
    # Local file header: version, flags, method, time, date, crc, compressed
    # size, uncompressed size, name length, extra length.
    (_, _, method, _, _, _, _, _, nlen, elen) = struct.unpack("<HHHHHIIIHH",
                                                              blob[4:30])
    name = blob[30:30 + nlen].decode()
    if not name.endswith("100d.txt"):
        raise RuntimeError(f"expected the 100d member first, found {name}")
    if method != 8:
        raise RuntimeError(f"expected deflate, found compression method {method}")

    data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(blob[30 + nlen + elen:])
    text = data.decode("utf8", "ignore")
    if prefix_only:                       # the last line is almost certainly cut
        text = text[:text.rindex("\n") + 1]
    with open(path, "w") as f:
        f.write(text)
    print(f"  wrote {path} ({text.count(chr(10))} words)", file=sys.stderr)


def load(path):
    words, rows = [], []
    for line in open(path):
        parts = line.rstrip("\n").split(" ")
        if len(parts) != DIM + 1:         # a truncated final line, or a stray
            continue
        words.append(parts[0])
        rows.append(parts[1:])
    V = np.array(rows, dtype=np.float32)
    return words, V, {w: i for i, w in enumerate(words)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", action="store_true",
                    help="fetch only the most frequent ~41.5k words (14 MB)")
    ap.add_argument("--cache", default=CACHE, help="where to keep the vectors")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        fetch(args.cache, args.prefix)
    words, V, idx = load(args.cache)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    print(f"\nvocabulary {len(words)}, {V.shape[1]} dimensions\n")

    for a, b, c, expected in QUERIES:
        q = V[idx[a]] - V[idx[b]] + V[idx[c]]
        q /= np.linalg.norm(q)
        sims = Vn @ q
        order = np.argsort(-sims)[:8]
        print(f"{a} - {b} + {c}   (the analogy expects {expected})")
        for rank, i in enumerate(order, 1):
            note = "   <- a query word" if words[i] in (a, b, c) else ""
            print(f"  {rank}. {words[i]:<12} {sims[i]:.3f}{note}")
        print(f"  cosine with {expected}: {float(Vn[idx[expected]] @ q):.3f}\n")

    print("the offset that is supposed to be one offset:")
    for a, b, c, d in OFFSETS:
        u, v = V[idx[a]] - V[idx[b]], V[idx[c]] - V[idx[d]]
        cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
        print(f"  cos( {a}-{b} , {c}-{d} ) = {cos:.3f}")

    def unit_offsets(pairs):
        out = []
        for a, b in pairs:
            if a in idx and b in idx:
                v = V[idx[a]] - V[idx[b]]
                out.append(v / np.linalg.norm(v))
        return out

    def spread(offs):
        cs = [float(u @ v) for u, v in itertools.combinations(offs, 2)]
        return float(np.mean(cs)), min(cs), max(cs)

    print("\nhow well each family shares one direction:")
    for name, pairs in FAMILIES.items():
        m, lo, hi = spread(unit_offsets(pairs))
        print(f"  {name:<20} n={len(pairs):2d}  mean {m:.3f}   ({lo:.3f} to {hi:.3f})")

    # The control. Without it the numbers above have no scale: 0.27 could be a
    # lot or nothing, and only the noise floor says which.
    rng = random.Random(0)
    common = [w for w in words[:20000] if w.isalpha()]
    control = [(rng.choice(common), rng.choice(common)) for _ in range(40)]
    m, lo, hi = spread(unit_offsets(control))
    print(f"  {'unrelated pairs':<20} n=40  mean {m:.3f}   ({lo:.3f} to {hi:.3f})")


if __name__ == "__main__":
    main()
