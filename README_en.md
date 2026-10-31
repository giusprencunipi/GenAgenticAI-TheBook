# Generative and Agentic AI: Algorithms and Architectures — companion

*[Versione italiana](README_it.md)*

This repository is the companion to the book. The book is a single argument,
read straight through; what a printed page cannot carry lives here.

**The repository is being prepared.** For now it holds the companion volume,
the code and the supporting figures; the rest will follow when the book is
published.

The **companion volume** is [`companion-volume.pdf`](companion-volume.pdf),
in English: it holds the exercises with their solutions, computations in full
and a dated chapter on current practice. It is updated when needed; the copy
here is always the current one.

## What is here

- **Exercises and solutions** — in the companion volume, for the book's technical chapters,
  from Autoencoders to Agentic AI. They are not printed in the book on purpose:
  nothing in its argument depends on them. The book is complete without a single
  exercise attempted; these are an instrument for whoever wants one.

- **Current practice** — a dated chapter of the companion volume, with the
  numbers and choices the book does not print because they would date: how
  big models are, how they are adapted and served, what a reasoning model
  spends.

- **Computations in full** — in the companion volume, the arithmetic the book states a result for rather
  than spreading over three pages, such as the exhaustive scan behind the claim
  about how far positional similarity decays.

- **Code** — the scripts behind the book's figures and numbers: every figure
  that was computed rather than drawn, and every number that came from a
  measurement, names the file that produced it. Among them are the scripts
  behind the figures that came from a model actually trained: the manifold of
  Chapter 4, the mode collapse of Chapter 5, the corrupted digits of
  Chapter 6. Seeded, so that a sceptical reader can regenerate them. They are
  in [`supportScripts-repo/`](supportScripts-repo/).

- **Supporting figures** — figures that go with the book but are not printed
  in it, such as the 64 first-layer AlexNet filters in colour from Appendix D.
  They are in [`supportFigures-repo/`](supportFigures-repo/).

- **Corrections and updates.** The material on language models and on agents
  has the shorter half-life, and what a printed book cannot revise, a
  repository can.

## What is published here

Compiled PDFs, and code. The LaTeX sources of the book and of this companion
are not part of this repository.

## Licence

Everything in this repository — PDFs, text and code — is published under
**[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International](https://creativecommons.org/licenses/by-nc-nd/4.0/)**
(CC BY-NC-ND 4.0). The full text is in [LICENSE](LICENSE).

In short: you may download and redistribute the material, whole and unmodified,
with attribution; you may not use it commercially; and you may not distribute
modified or adapted versions of it.

For uses this licence does not allow — adapting the teaching material for a
course, in particular — write to the authors.

## Reporting an error

Corrections, better implementations and counterexamples are welcome —
counterexamples particularly. Open an issue.
