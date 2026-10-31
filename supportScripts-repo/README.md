# supportScripts-repo

Every figure and every number in the book *Generative and Agentic AI:
Algorithms and Architectures* that came from a computation rather than from an
author's judgement is produced by one of the scripts here: they are the code the
book points at when it says that something can be re-run.

Run them from the root of this repository, not from inside this folder. Each
writes its figures either next to the script or in the current directory:

    python3 supportScripts-repo/make_mnist_figures.py

| Script | Produces | Used by |
|---|---|---|
| `make_ae_bottleneck_numbers.py` | the reconstruction error at five widths of the bottleneck | Chapter 3 |
| `make_ae_pretraining_numbers.py` | the table of test accuracies in the few-label pre-training example | Chapter 3 |
| `make_mnist_three.py` | `mnist_three.png`, the single digit followed through the worked example | Chapters 3 and 4 |
| `make_mnist_figures.py` | `mnist_manifold.png`, `mnist_interp_17.png`, `mnist_interp_38.png` | Chapter 4 |
| `make_vae_failure_modes.py` | the codes plotted in the figure comparing the autoencoder, the encoder without KL term and the VAE | Chapter 4 |
| `make_flow_chain.py` | the points of the figure showing a chain of invertible transformations (normalising flows) | Chapter 4 |
| `make_gan_figures.py` | `gan_mnist_samples.png`, `gan_mnist_vae_samples.png`, and the mode-collapse coordinates | Chapter 5 |
| `make_diffusion_figure.py` | `forward_process.png` | Chapter 6 |
| `make_gaussian_sample_stats.py` | the 10,000-draw column of the table in the example on sampling from the standard Gaussian | Chapter 6 |
| `make_posenc_decay.py` | the decay of positional-encoding similarity: monotone window and worst inversion for eight model widths | Chapter 7 |
| `make_filter_figure.py` | `conv1_filters.png`, and `conv1_filters_colour.png` in `supportFigures-repo/` | Appendix D |
| `make_analogy_numbers.py` | the analogy and subspace figures quoted in the text | Appendix E |
| `make_tsne_figure.py` | `tsne_seeds.png` and the correlations quoted in the text | Appendix E |

Four of them train a model and take between one and about eight minutes; the
rest are arithmetic and return immediately. All are seeded, so a second run reproduces the first.

Three download something on first use and cache it: MNIST (a few MB), the
GloVe 6B vectors (134 MB, and `make_analogy_numbers.py --prefix` cuts that to
14 MB with identical results), and the AlexNet weights (244 MB). None of those
belong in version control, and the repository's `.gitignore` excludes them.

A note on the ones that print numbers rather than pictures. Where the book
quotes a correlation or a count, that value is computed here and printed by the
script, so that the text and the computation cannot drift apart. If a number in
the book disagrees with what the script prints, the script is right.

Dependencies: numpy and matplotlib throughout; torch for the five that need a
trained network, and torchvision as well for the one that loads AlexNet.
`make_ae_bottleneck_numbers.py` imports its network and data loader from
`make_ae_pretraining_numbers.py`, so the two must stay side by side. Nothing else — the t-SNE in
`make_tsne_figure.py` is implemented rather than imported, which keeps the
dependency list short and, in a book about opening up mechanisms, seemed the
consistent thing to do.
