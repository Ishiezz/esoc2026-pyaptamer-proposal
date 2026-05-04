# ESoC 2026 — Project Application: Isha Singh
## Project: *Adding the AptaDiff Algorithm* — `gc-os-ai/pyaptamer`

---

| | |
|---|---|
| **Applicant** | Isha Singh |
| **GitHub** | [@Ishiezz](https://github.com/Ishiezz) |
| **Email** | singhishaa.24@gmail.com |
| **Institution** | Newton School of Technology, ADYPU, Pune — B.Tech AI & ML, Year 2 |
| **Location** | India (IST, UTC+5:30) |
| **Overlap with mentors** | 12:30–18:30 IST = 09:00–15:00 CET |
| **Project choice** | Adding the AptaDiff Algorithm (Hard, 300 hours) |
| **Application deadline** | April 30, 2026 — 18:00 UTC |

---

> **TL;DR: Why select me for AptaDiff?**
> - **I have already built the exact architecture you need.** In PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), I implemented the sklearn-compatible `fit()` and batch inference pipeline over a PyTorch Lightning backbone for `AptaTrans` — the exact same integration pattern the project card demands for AptaDiff.
> - **I have read the AptaDiff paper and reference code in depth.** I understand that AptaDiff is a two-stage VAE + Multinomial Diffusion model — not a simple DDPM — and I have mapped every component to its pyaptamer equivalent before writing a single line of proposal. My implementation plan is based on what the code actually does, not what the abstract says.
> - **I have already hardened the data pipeline AptaDiff depends on.** PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632), and [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) fix bugs and maintenance issues across `encode_rna`, `filter_words`, `GreedyEncoder`, and `BaseTransform` — the exact components AptaDiff's data layer will use. I found all four independently by reading source code, not open issues.

---

## 1. Why This Project Matters — and Why I Am Ready to Build It

AptaNet and AptaTrans answer: *"Does this aptamer bind this protein?"* AptaDiff inverts the question: *"Given this protein, what aptamers should I generate?"* Adding AptaDiff completes the in-silico SELEX cycle in Python:

```
generate candidates → AptaDiff
        ↓
score binding probability → AptaNet / AptaTrans
        ↓
filter top candidates → wet lab
```

This is the primary roadmap goal identified in Issue [#81](https://github.com/gc-os-ai/pyaptamer/issues/81). Nobody has claimed it. I am uniquely positioned to deliver it — not because I have the most experience, but because I have already done the hard preparatory work that most applicants skip.

In April–May 2026, I made **six substantive contributions** to `gc-os-ai/pyaptamer`: I implemented the full sklearn-compatible `fit()` / `predict_interactions()` interface for `AptaTransPipeline` (PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)), independently audited the codebase to find and fix three unreported bugs in the data pipeline (PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)), and fixed a maintenance issue in `BaseTransform` that had been present since the class was created (PR [#637](https://github.com/gc-os-ai/pyaptamer/pull/637)). I did not find these issues by reading open issues. I found them by reading source code.

---

## 2. Why I Am the Right Candidate

### 2.1 I Have Already Done the Core Pattern — Mapped to Every AptaDiff Requirement

| AptaDiff requirement | My existing evidence |
|---|---|
| sklearn-style `fit()` API | `AptaTransPipeline.fit(X, y)` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| Batch inference via Lightning `Trainer` | `predict_interactions()` via `Trainer.predict()` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| `_prepare_dataloader` / `APIDataset` integration | Used and extended in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| Sequence tokenization (`encode_rna`, `GreedyEncoder`) | Audited, hardened, tested — PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |
| Vocabulary pipeline (`filter_words`) | Hardened with guard + test — PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630) |
| Base class architecture (`BaseTransform`) | Abstract method convention fixed — PR [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) |
| Input validation + error handling | `ValueError` / `NotImplementedError` guards across PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632), [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) |
| Test coverage | 9 tests in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); first-ever 7-test suite for `GreedyEncoder` in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |

### 2.2 Microsoft VS Code — Production Open Source Track Record

> **PR [#281302](https://github.com/microsoft/vscode/pull/281302) merged into Microsoft VS Code (184k+ ⭐)**

In January 2026, I fixed a semver comparison bug in the extension linter — a production TypeScript codebase with an extremely high bar for correctness and review. This is evidence that I can ship code that passes senior engineering review in large, unfamiliar codebases.

### 2.3 Technical Skills — Direct Match

| Skill | Evidence |
|---|---|
| Python (advanced) | PRs [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632), [#637](https://github.com/gc-os-ai/pyaptamer/pull/637); my project ContraLegal-AI (97%+ accuracy ML system) |
| PyTorch | `AptaTransPipeline` training loop; MCTS fix — PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495) |
| PyTorch Lightning | `AptaTransLightning.predict_step()` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); Apple Silicon MPS compatibility fix |
| scikit-learn interface | `fit()`, `predict_interactions()`, `score()` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| Diffusion models | Read AptaDiff paper + reference code in full depth (see Section 3); trained transformer + attention architectures in my project ContraLegal-AI; studied Ho et al. DDPM (2020) and Hoogeboom et al. Multinomial Diffusion (2021) |
| VAE architecture | Studied 1D CNN encoder + profile HMM decoder in AptaDiff reference; implemented autoencoder components in my project ContraLegal-AI |
| Testing (pytest) | 9 tests — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); 20 tests after PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617); 4 tests after PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630); first-ever 7-test suite for `GreedyEncoder` — PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |

---

## 3. What I Found When I Actually Read the AptaDiff Code

### 3.1 The Real Architecture — Two Stages, Not One

The AptaDiff proposal summary says "conditional diffusion model." That is true but incomplete. After reading the paper and reference code at `wz-create/AptaDiff`, the actual architecture is:

**Stage 1 — VAE with profile HMM decoder:**
- A 1D CNN encoder maps aptamer sequences into a 2D motif-dependent latent space `z`
- The decoder is a **profile HMM** (not a standard MLP or transformer), which gives the latent space its motif structure
- Training: ELBO loss = reconstruction + KL divergence
- Output: `z ∈ ℝ²` — a 2D coordinate per aptamer sequence

**Stage 2 — Multinomial Diffusion conditioned on `z`:**
- This is **not Gaussian DDPM**. It is Multinomial Diffusion (Hoogeboom et al., 2021) for categorical data
- Each nucleotide is one-hot encoded: A=(1,0,0,0), C=(0,1,0,0), G=(0,0,1,0), T=(0,0,0,1)
- Forward process gradually replaces nucleotides with uniform noise over K=4 classes
- The denoiser is a transformer (`dim=512, heads=16, depth=12`) that takes `(x_t, z, t)` and predicts `x_0`
- T=1000 diffusion steps, Adam optimizer, batch size 64, trained up to 1000 epochs

**Stage 3 — Bayesian Optimization for affinity guidance:**
- A GMM clusters the 2D latent space
- Bayesian optimization searches the latent space guided by SPR-measured binding affinity
- This is how protein-specificity enters — not through direct protein conditioning, but through Bayesian optimization over a protein-specific latent space

### 3.2 Five Concrete Integration Decisions I Have Already Made

**1. DNA → RNA conversion is already solved.** The reference uses DNA (A/C/G/T). pyaptamer uses RNA. The `dna2rna()` function in `_rna.py` — which I audited in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) — handles this exactly.

**2. The VAE is the hard part, not the diffusion.** The profile HMM decoder is non-trivial. My plan: implement a simplified CNN decoder first, validate latent space quality, and treat the profile HMM as an optional enhancement.

**3. `prot_words` is not used for aptamer sequences.** The reference uses raw one-hot encoding for nucleotides. I will add a new `AptaDiffDataset` for raw one-hot encoding, reusing `filter_words` (hardened in PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)) only for the Bayesian optimization affinity data.

**4. The `generate()` interface needs two modes.** Based on the reference: `generate(mode="gmm")` for motif-diverse sampling and `generate(mode="bo", affinity_data=...)` for affinity-optimised generation.

**5. The noise schedule is linear in the reference.** I will implement cosine schedule as the default (better for shorter biological sequences) with linear as a fallback, exposed as a `noise_schedule` parameter.

### 3.3 The Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ AptaDiffPipeline (sklearn-style)                            │
│  ├── fit(X_seqs)          → trains VAE + diffusion backbone │
│  └── generate(n, mode)    → returns candidate aptamers      │
│         │                                                   │
│         ├── AptaDiffVAE (Stage 1)                           │
│         │     ├── CNN encoder: seq → z ∈ ℝ²                 │
│         │     └── CNN/HMM decoder: z → seq reconstruction   │
│         │                                                   │
│         └── AptaDiffLightning (Stage 2, pl.LightningModule) │
│               ├── training_step: Multinomial diffusion loss  │
│               ├── predict_step: reverse denoising loop       │  ← same pattern as PR #493
│               └── AptaDiffDataset                           │
│                     └── dna2rna() → one-hot → DataLoader    │  ← uses _rna.py (PR #617)
│                                                             │
│  ├── AptaDiffBO (Stage 3, optional)                         │
│        └── GMM → Bayesian opt → affinity-guided generation  │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Plan

### The Three Hardest Technical Challenges

**Challenge 1: Implementing the VAE (Weeks 1–3)**

The profile HMM decoder is the most complex component. My approach: implement a 1D CNN decoder first, validate latent space quality using GMM clustering, then add the HMM decoder as a configurable option if time permits. I will be transparent with mentors about this tradeoff from day one.

**Challenge 2: Multinomial Diffusion over discrete sequences (Weeks 3–6)**

The key difference from Gaussian: the forward process adds uniform noise over K classes rather than Gaussian noise, and the reverse process predicts `x_0` via a categorical posterior. The loss is KL divergence between the true posterior `q(x_{t-1}|x_t, x_0)` and the learned `p(x_{t-1}|x_t)`. My plan: implement `q_sample`, validate it empirically by confirming `q(x_T | x_0)` converges to uniform over A/C/G/U, then build the transformer denoiser and training loop.

**Challenge 3: Clean sklearn API over a two-stage model (Weeks 7–9)**

`fit()` must train Stage 1 (VAE) and Stage 2 (diffusion) sequentially. My solution: `generate(n, mode="gmm")` and `generate(n, mode="bo", affinity_scores=...)`, documented clearly, with sensible defaults.

### Milestone Map

| Milestone | Deliverable | When |
|---|---|---|
| **M1** | `AptaDiffVAE` passes overfit test on 100 sequences; latent space shows cluster structure | End of Week 3 |
| **M2** | Multinomial diffusion forward + reverse validated; `AptaDiffLightning` trains to convergence | End of Week 6 |
| **M3** | `AptaDiffPipeline.fit()` + `generate(mode="gmm")` end-to-end; sequences pass validity checks | End of Week 9 |
| **M4** | Full test suite; `notebooks/aptadiff_tutorial.ipynb`; BO mode; docs; PR ready for merge | End of Week 12 |

### What I Will Ask Mentors to Review Early

1. The VAE decoder choice (CNN vs HMM) — I want explicit agreement before I invest time
2. The `generate()` API signature — naming decisions are hard to undo
3. My `q_sample` implementation before building the full training loop

---

## 5. Risks & Mitigation

| Risk | Mitigation |
|---|---|
| **Profile HMM decoder complexity** | Ship CNN decoder first; HMM decoder is a configurable enhancement, not a blocker |
| **Multinomial diffusion subtleties** | Validate `q(x_T\|x_0)` → uniform empirically before building reverse process |
| **API Drift** — PR #441 still stabilizing | Build `AptaDiffDataset` as standalone with minimal coupling to `APIDataset` |
| **Silent data failures** | `ValueError` / `NotImplementedError` guard pattern already established across PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632), [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) |

---

## 6. Deliverables

| Deliverable | Maps to Project Card | Status |
|---|---|---|
| `AptaDiffVAE` — CNN encoder + decoder, ELBO training | "adapting AptaDiff from scratch" | Week 1–3 |
| `AptaDiffLightning` — Multinomial diffusion training + sampling | "adapting AptaDiff from scratch" | Week 3–6 |
| `AptaDiffPipeline.fit()` + `generate()` | "scikit-learn-style API" | Week 7–9 |
| Full test suite (shape, type, device, integration) | "write tests" | Ongoing |
| `notebooks/aptadiff_tutorial.ipynb` | "notebook demonstrating public API" | Week 10–11 |
| Hugging Face Spaces demo | Stretch goal | Week 10 |
| **PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)** — AptaTrans `fit()` + `predict_interactions()` | Prerequisite — sklearn API pattern | ✅ Open |
| **PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495)** — MCTS phantom visit fix | Prerequisite — algorithmic correctness | ✅ Closed (duplicate of #484 — independently correct fix) |
| **PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617)** — `encode_rna` docstring + `return_type` validation | Bug fix — RNA tokenizer used in `AptaDiffDataset` | ✅ Open |
| **PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)** — `filter_words` empty dict fix | Bug fix — vocabulary pipeline used in BO stage | ✅ Open |
| **PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)** — `GreedyEncoder` crash fix + first test suite | Bug fix — sequence encoder used in data layer | ✅ Open |
| **PR [#637](https://github.com/gc-os-ai/pyaptamer/pull/637)** — `BaseTransform` abstract method fix + module attributes | Maintenance — base class used by all transformers | ✅ Open |

---

## 7. My Contribution Journey

### 7.1 How I Found the Real Work (April 19–20)

I forked `gc-os-ai/pyaptamer`, ran the full test suite, and read every open issue. Most were claimed within hours. My strategy: look past the easy wins and find the work requiring genuine codebase depth. The most important item was **[Issue #190](https://github.com/gc-os-ai/pyaptamer/issues/190)** — filed by maintainer Satvik in October 2025, open six months, one earlier incomplete attempt — which is the exact API pattern the AptaDiff project card requires.

### 7.2 Collaborating on Architecture Before Writing Code (Draft PR #447)

Before implementing `fit()`, I opened [PR #447](https://github.com/gc-os-ai/pyaptamer/pull/447) as a draft. Contributor `siddharth7113` advised me to build on PR #441's `APIDataset` refactor. Instead of pushing conflicting code, I left #447 as a draft and **read all 21 commits of PR #441** before writing a line of PR #493.

### 7.3 PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495) — MCTS Algorithmic Correctness Fix

Found a subtle UCT math bug: `TreeNode.__init__` set `n_visits = 1` instead of `0`, destroying the infinite exploration bonus for unvisited nodes. Closed as duplicate of PR #484 — I had arrived at the identical correct fix independently. Commented professionally and moved on to find harder work.

### 7.4 PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) — AptaTrans `fit()` + `predict_interactions()`

Complete, production-quality implementation of Issue #190: `fit(X, y)`, `predict_interactions(X)`, `AptaTransLightning.predict_step()`, `prot_words` docstring standardization, return type documentation, and 9 new tests covering input shapes, output types, device handling, and Apple Silicon MPS.

### 7.5 PR [#494](https://github.com/gc-os-ai/pyaptamer/pull/494) — Closed, and What I Learned

*(Minor `AptaNetPipeline.fit()` fix — closed by Satvik as a parallel PR addressed it. Reinforced my focus on deeper architectural work.)*

### 7.6 PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — Independent Bug Audit

After closing PR #495, I read the components AptaDiff would depend on and wrote reproduction scripts to test edge cases. Found three bugs nobody had reported:

- **PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617):** `encode_rna` docstring fully copy-pasted from protein encoder (wrong terminology + typo "trunacted"); invalid `return_type` silently returned tensor instead of `ValueError`. All 20 tests pass.
- **PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630):** `filter_words` emitted silent `RuntimeWarning` on empty dict — dangerous because it's called directly in `AptaTransPipeline._init_words()`. All 4 tests pass.
- **PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632):** `GreedyEncoder` crashed with confusing `ValueError: max() iterable argument is empty` on `words={}`, or silently returned all-zeros when `word_max_len` was set. Created the first-ever test suite for `GreedyEncoder` — 7 tests. All pass.

### 7.7 PR [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) — BaseTransform Maintenance

Continued auditing and found a maintenance issue in `pyaptamer/trafos/base/_base.py` that had existed since the class was created in PR #189 (November 2025). Three abstract methods — `_fit`, `_transform`, and `_transform_element` — raised `ValueError` instead of the correct `NotImplementedError`. Using `ValueError` is misleading (suggests bad input rather than missing implementation) and inconsistent with `skbase.BaseEstimator` which `BaseTransform` inherits from. Also added missing `__author__` and `__all__` module-level attributes that every other module in pyaptamer has. Single-file change, zero test changes needed — the existing `GreedyEncoder` test suite exercises `BaseTransform` through inheritance.

---

## 8. About Me

When I started on pyaptamer, the first thing I did was clone the repo and run the tests — not read the issues. One doctest was silently passing because it was malformed. Nobody had noticed. I fixed it as part of PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493). That is how I work.

I am a second-year B.Tech student in AI & ML at Newton School of Technology, Pune. I have a merged PR in Microsoft VS Code (184k+ ⭐). I have six open PRs in pyaptamer touching the sklearn API layer, the algorithmic core, the data pipeline, and the base class architecture. I built my project ContraLegal-AI — a BERT + RAG legal risk classifier at 97%+ accuracy — entirely from scratch. I am not the most experienced candidate in this pool. But I am the one who independently hardened the four data pipeline components that AptaDiff will depend on, read the AptaDiff paper's full Methods section and reference implementation before writing this proposal, and named the five specific integration decisions I have already made and why.

---

## 9. Availability & Commitment

| Period | Availability |
|---|---|
| June 9 — September 2026 | Full-time, 35–40 hours/week |
| Exam period (if any) | Will communicate 2+ weeks in advance |

**Time zone:** IST (UTC+5:30) — **6 hours daily overlap** with European mentors (12:30–18:30 IST = 09:00–15:00 CET)  
**Communication:** GitHub, Discord, email — responsive within same working day

---

## 10. Links

| Item | Link |
|---|---|
| GitHub Profile | https://github.com/Ishiezz |
| Fork | https://github.com/Ishiezz/pyaptamer |
| **PR #493** — AptaTrans `fit()` + `predict_interactions()` | https://github.com/gc-os-ai/pyaptamer/pull/493 |
| **PR #495** — MCTS phantom visit fix (closed — duplicate of #484) | https://github.com/gc-os-ai/pyaptamer/pull/495 |
| **PR #617** — `encode_rna` docstring fix + `return_type` validation | https://github.com/gc-os-ai/pyaptamer/pull/617 |
| **PR #630** — `filter_words` empty dict `RuntimeWarning` fix | https://github.com/gc-os-ai/pyaptamer/pull/630 |
| **PR #632** — `GreedyEncoder` crash fix + first-ever test suite | https://github.com/gc-os-ai/pyaptamer/pull/632 |
| **PR #637** — `BaseTransform` abstract method fix + module attributes | https://github.com/gc-os-ai/pyaptamer/pull/637 |
| PR #494 — AptaNet fix (closed by Satvik) | https://github.com/gc-os-ai/pyaptamer/pull/494 |
| PR #447 — Draft (superseded by #493) | https://github.com/gc-os-ai/pyaptamer/pull/447 |
| **Issue #190** — AptaTrans sklearn API | https://github.com/gc-os-ai/pyaptamer/issues/190 |
| **Issue #81** — Algorithm wishlist (AptaDiff) | https://github.com/gc-os-ai/pyaptamer/issues/81 |
| Issue #483 — MCTS phantom visit | https://github.com/gc-os-ai/pyaptamer/issues/483 |
| Issue #616 — `encode_rna` bug | https://github.com/gc-os-ai/pyaptamer/issues/616 |
| PR #441 — APIDataset refactor | https://github.com/gc-os-ai/pyaptamer/pull/441 |
| **Microsoft VS Code PR #281302** (merged) | https://github.com/microsoft/vscode/pull/281302 |
| AptaDiff paper (Briefings in Bioinformatics, 2024) | https://academic.oup.com/bib/article/25/6/bbae517/7828722 |
| AptaDiff reference code | https://github.com/wz-create/AptaDiff |
| ESoC 2026 pyaptamer project card | https://github.com/european-summer-of-code/esoc2026/blob/main/cards/pyaptamer.md |

---

*Submitted: April 30, 2026 — Batch 2 deadline*
*Updated: May 4, 2026 — PR #637 (BaseTransform MNT) added*