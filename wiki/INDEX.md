# Wiki

Internal knowledge base for the raw-audio research project. **Internal** — for skills, future Claude sessions, and the user reading notes between sessions. User-facing summaries belong in `reports/` as HTML, not here.

## Structure

- `wiki/INDEX.md` — this file.
- `wiki/concepts/` — cross-paper concept pages. Each page synthesises what we've learned about a broad topic across all papers that touch it.

The wiki is **a small set of dense concept pages, not a sprawl of narrow ones**. New pages are added when a topic has accumulated enough cross-paper content across the wiki to fill its own dense page — i.e. the split is consolidating scattered material, not hosting a single paper. Most new material lives as a subsection of an existing page until that bar is reached.

## Per-paper notes live elsewhere

Per-paper writeups go in `papers/`. The wiki links to them but doesn't duplicate them. If you want the full method breakdown of a single paper, read `papers/<slug>.md`.

## Living-documents principle

Pages here **evolve continuously**. Whenever a new paper, experiment, or observation sharpens our understanding — a comparison emerges, a contradiction surfaces, a framing improves — the relevant concept page is edited.

**Lean and refined.** This wiki is a synthesis surface, not a paper dump. Prune as much as you add. If a section has grown stale or unfocused, rewrite it. If two pages overlap too much, merge them.

## What goes in a concept page

- A one-sentence purpose statement at the top — what question does this page answer?
- Short prose synthesis of what's currently known.
- Subsections / comparison tables where multiple methods can be tabled side-by-side.
- Links to relevant `papers/<slug>.md` writeups.
- Open questions / things we'd like to test.
- A `last reviewed: YYYY-MM-DD` line.

If a page's shape doesn't fit, the shape follows the content.

## Concept pages

- [methods](concepts/methods.md) — **from-scratch** training paradigms: score-based diffusion, flow matching, MeanFlow family (Shortcut → MeanFlow → iMF / SoFlow / pMF / AlphaFlow / SplitMeanFlow / Int-MeanFlow / Variance-Reduction MF / TVM), Drifting / WGF-then-compress (Drifting / Sinkhorn-Drifting / W-Flow / Gradient-Flow Drifting), Transition Models, EqM, IMM, consistency-from-scratch, SSL pre-training, from-scratch GAN (HiFi-GAN, R3GAN, ComVo), scoring-rule (energy / Laplace / signature kernel), **masked-token / discrete-AR with bit modelling (BAR — `L × (1 + N)` two-loop inference, contrastive data point)**. The "how do you train it from random init?" page.
- [post-training](concepts/post-training.md) — methods that require a pretrained generator: adversarial distillation (ADD/LADD/DMD2), adversarial post-training (Seaweed-APT, ARC), FM-pretrain + GAN finetune (Flow2GAN, PeriodWave-Turbo), score/trajectory distillation (SiD-DiT), FD-loss, representation distillation. The "given a working model, what next?" page.
- [architecture](concepts/architecture.md) — network shapes for sample-space audio: 1D conv U-Net, patched ViT, STFT-domain ConvNeXt. The "what's the network?" page.
- [inference](concepts/inference.md) — CPU latency target, NFE budgets, schedule × sampler × NFE coupling, GFLOPs Pareto front. The "how fast does it run?" page.
- [evaluation](concepts/evaluation.md) — FID/FAD and their fixes along three axes: embedding saturation (FDr^k, FD-as-loss), Gaussian closed-form (MIND / sliced-W), and single-metric fragility (Räisä sanity checks; Vendi / Coverage diversity bundle). The "how do we measure quality?" page.
- [representation-space](concepts/representation-space.md) — pixel/sample-space vs latent. The "why generate in raw sample space at all?" page — project-defining framing.
