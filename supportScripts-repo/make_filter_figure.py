#!/usr/bin/env python3
"""Regenerate the first-layer filter montage used by the appendix on convolutions.

    conv1_filters.png           the 44 luminance-dominated filters, greyscale
    supportFigures-repo/conv1_filters_colour.png
                                all 64, in colour, for the companion repository

    python3 supportScripts-repo/make_filter_figure.py

The appendix says that the filters a convolutional network learns in its first
layer are oriented edges, gratings and colour blobs, and that nobody put them
there. That is a claim about pictures, so the pictures had better be real. They
are: these are the 64 filters of the first layer of AlexNet as released, weights
and all, with no selection, no retouching and no training done here. The
counting is done by the script too, so the numbers in the caption cannot drift
away from the figure.

Two groups come out of that counting. For each filter, compare how much it
varies across the three colour channels with how much its luminance varies
across space. Forty-four vary mostly in luminance -- those are the edges and
the gratings -- and twenty vary mostly in colour, which are the opponent blobs.
The split is not a threshold anyone tuned; it falls out of the weights.

Only the first group can be printed. This book is produced with four fixed
inks, and a colour-opponent filter is not expressible in them, so the printed
figure is the 44 luminance filters in greyscale, which the ink budget does not
constrain. The colour montage is written alongside for the repository, where
there is no such limit.

The luminance filters are ordered by the orientation of their dominant spatial
frequency, taken from the peak of the 2-D Fourier transform. Nothing depends on
that ordering; it only makes the sweep through the angles visible, which is
otherwise lost in a random grid.

The weights download once to the torch hub cache (~244 MB). If Python has no CA
bundle -- the state a python.org install on macOS is in until told otherwise --
fetch them with curl first:

    U=$(python3 -c "from torchvision.models import AlexNet_Weights; \\
                    print(AlexNet_Weights.IMAGENET1K_V1.url)")
    mkdir -p ~/.cache/torch/hub/checkpoints
    curl -sSL -o ~/.cache/torch/hub/checkpoints/$(basename $U) "$U"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREY = "conv1_filters.png"
COLOUR = "supportFigures-repo/conv1_filters_colour.png"   # published, not printed
DPI = 300


def load_conv1():
    from torchvision.models import alexnet, AlexNet_Weights
    m = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    return m.features[0].weight.data.numpy()          # (64, 3, 11, 11)


def split(w):
    """Separate filters that vary mostly in luminance from those that vary
    mostly in colour. No threshold: whichever variation is larger decides."""
    flat = w.reshape(len(w), 3, -1)
    across_channels = flat.std(axis=1).mean(axis=1)
    across_space = flat.mean(axis=1).std(axis=1)
    luminance = across_channels <= across_space
    return np.where(luminance)[0], np.where(~luminance)[0]


def orientation(f):
    """Angle of the dominant spatial frequency, for ordering only."""
    g = f.mean(axis=0)
    g = g - g.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(g, s=(32, 32))))
    spectrum[16, 16] = 0                              # drop the DC term
    i, j = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return np.arctan2(i - 16, j - 16) % np.pi


def unit(f):
    f = f - f.min()
    return f / (f.max() + 1e-12)


def montage(path, tiles, rows, cols, grey):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.62, rows * 0.62))
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([])
        ax.set_yticks([])
        if k < len(tiles):
            for s in ax.spines.values():
                s.set_linewidth(0.4)
                s.set_color("0.6")
            if grey:
                ax.imshow(unit(tiles[k].mean(axis=0)), cmap="gray",
                          interpolation="nearest")
            else:
                ax.imshow(unit(tiles[k]).transpose(1, 2, 0),
                          interpolation="nearest")
        else:
            ax.axis("off")
    plt.subplots_adjust(wspace=0.12, hspace=0.12,
                        left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    w = load_conv1()
    lum, col = split(w)
    print(f"first layer: {len(w)} filters of {w.shape[2]}x{w.shape[3]}, "
          f"{w.shape[1]} channels")
    print(f"  {len(lum)} vary mostly in luminance  -> {GREY}")
    print(f"  {len(col)} vary mostly in colour")
    print(f"  all {len(w)}                        -> {COLOUR}")

    order = sorted(lum, key=lambda i: orientation(w[i]))
    montage(GREY, [w[i] for i in order], 4, 11, grey=True)
    montage(COLOUR, [w[i] for i in list(order) + list(col)], 8, 8, grey=False)


if __name__ == "__main__":
    main()
