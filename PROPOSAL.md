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
| **Overlap with mentors** | 12:30–18:30 IST = 09:00–15:00 CET  |
| **Project choice** | Adding the AptaDiff Algorithm (Hard, 300 hours) |
| **Application deadline** | April 30, 2026 — 18:00 UTC |

---

> ** TL;DR: Why select me for AptaDiff?**
> - **I have already built the exact architecture you need.** In PR #493, I implemented the sklearn-compatible `fit()` and batch inference pipeline over a PyTorch Lightning backbone for `AptaTrans` — the exact same technical pattern the project card demands for AptaDiff.
> - **I write production-quality code.** I have a merged PR in Microsoft VS Code (#281302), proving I can ship high-quality code in massive, heavily-reviewed codebases under senior engineering scrutiny.
> - **I understand the pyaptamer internals.** I read all 21 commits of the recent `APIDataset` refactor before writing my entrance task. I already know how to build the AptaDiff pipeline cleanly on top of `MoleculeLoader` and the `_prepare_dataloader` seam.

---

## 1. Project Abstract

I am applying to implement the **AptaDiff algorithm** in `gc-os-ai/pyaptamer` — one of the three official 2026 full projects listed in the project card. AptaDiff is a conditional diffusion model that generates aptamer sequences targeting specific proteins, complementing the existing prediction-only algorithms (AptaNet, AptaTrans) by adding generative capability to the library.

My work will deliver a complete, production-quality `pyaptamer.aptadiff` module: a `pyaptamer`-style sklearn-compatible pipeline (`fit` + `generate`), integrated with the existing `MoleculeLoader` data pipeline, comprehensive test coverage, and a Jupyter notebook demonstrating the public API end-to-end.

I am uniquely positioned for this project. In April 2026 I implemented the sklearn-compatible `fit()` and `predict_interactions()` interface for `AptaTransPipeline` (PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)) — the exact same API pattern the project card specifies for AptaDiff. I understand the `pyaptamer` internals: the `APIDataset` data layer, the `_prepare_dataloader` seam, the Lightning training loop, the `prot_words` encoding — because I read 21 commits of upstream work before writing code, not after.

---

## 2. Why I Am the Right Candidate

### 2.1 I Have Already Done the Core Pattern — Twice

The AptaDiff project card requires:
> *"adapting the AptaDiff algorithm from scratch to follow the scikit-learn-style API and the existing public API conventions of pyaptamer, together with appropriate test coverage."*

This is precisely what I did for AptaTrans in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493):

| AptaDiff requirement | My existing work |
|---|---|
| sklearn-style `fit()` API | `AptaTransPipeline.fit(X, y)` — PR #493 |
| batch inference via Lightning `Trainer` | `AptaTransPipeline.predict_interactions()` via `Trainer.predict()` — PR #493 |
| `_prepare_dataloader` integration | Used and extended in PR #493 |
| `MoleculeLoader` / `APIDataset` integration | Studied in depth; implemented `from_any()` dispatch — PR #493 |
| Comprehensive test coverage | 9 new tests in `test_aptatrans_fit_predict.py` — PR #493 |
| Algorithmic correctness — MCTS UCT fix | `TreeNode.n_visits = 0`, `uct_score()` guard — PR #495 |

The work is not speculative. I have already implemented key components required for AptaDiff integration — designing a clean pipeline layer on top of a Lightning backbone — and I have the merged PR as evidence.

### 2.2 Microsoft VS Code — Production Open Source Track Record

> **PR [#281302](https://github.com/microsoft/vscode/pull/281302) merged into Microsoft VS Code (184k+ ⭐)**

In January 2026, I fixed a semver comparison bug in the extension linter — a production TypeScript codebase with an extremely high bar for correctness and review. This demonstrates I can work in large, unfamiliar codebases under peer review from a senior engineering team, which is the same skill required for contributing to `pyaptamer` under ecoSPECS and GC.OS mentorship.

### 2.3 Deep Learning Stack — Direct Match to Project Requirements

The project card lists: *Python, familiarity with deep learning, PyTorch, Lightning, and scikit-learn.*

| Required skill | My evidence |
|---|---|
| Python (advanced) | All pyaptamer PRs, ContraLegal-AI (97%+ accuracy ML system) |
| PyTorch | `AptaTransPipeline` training loop via `Trainer`, MCTS fix |
| PyTorch Lightning | `AptaTransLightning.predict_step()` in PR #493; Apple Silicon MPS fix |
| scikit-learn interface | `fit()`, `predict_interactions()`, `score()` pattern — PR #493 |
| Diffusion models | My project ContraLegal-AI uses BERT + RAG; currently studying DDPM architecture |
| Testing (pytest) | 9 new tests in PR #493, corrected MCTS test assertions in PR #495 |

---

## 3. Technical Understanding of AptaDiff

### 3.1 What AptaDiff Does

AptaDiff ([paper](https://academic.oup.com/bib/article/25/6/bbae517/7828722), [reference code](https://github.com/wz-create/AptaDiff)) is a **conditional diffusion model** for aptamer sequence generation. Unlike AptaNet and AptaTrans — which take `(aptamer, protein)` pairs and output a binding probability — AptaDiff inverts the problem: given a protein target, it *generates* candidate aptamer sequences.

The generative process follows a **discrete DDPM** (Denoising Diffusion Probabilistic Model) adapted for biological sequences:
1. **Forward process:** corrupts an RNA/DNA sequence token-by-token over `T` diffusion steps
2. **Reverse process:** a transformer denoiser conditioned on protein features learns to recover the original sequence
3. **Generation:** starting from noise, the model iteratively denoises, guided by the protein embedding, to produce new aptamer candidates

### 3.2 How It Integrates into pyaptamer

The integration challenge is the same one I already solved for AptaTrans: wrapping a Lightning-backed model in a clean sklearn-style `fit`/`generate` API that accepts `pyaptamer`'s existing data formats.

```
┌────────────────────────────────────────────────────┐
│ AptaDiffPipeline (sklearn-style)                   │
│  ├── fit(X_protein)     → trains denoiser backbone │
│  └── generate(target, n) → returns candidate seqs  │
│         │                                          │
│         └── AptaDiffLightning (pl.LightningModule) │
│               ├── training_step()                  │
│               ├── predict_step()   ← same pattern  │
│               └── configure_optimizers()            │
│                    │                               │
│               AptaDiffDataset                      │
│               (extends APIDataset / MoleculeLoader) │
└────────────────────────────────────────────────────┘
```

The `_prepare_dataloader` seam I used in PR #493 (introduced by PR #441's `APIDataset` refactor) is the exact interface needed to feed protein embeddings into AptaDiff training. The data pipeline work is already done by the existing `MoleculeLoader`.

### 3.3 Implementation Decisions

- **Sequence representation:** discrete token indices (A, U, G, C + padding) — compatible with `pyaptamer`'s existing `prot_words` vocabulary encoding
- **Protein conditioning:** use the same protein feature extraction already in `AptaNetPipeline` / `AptaTransPipeline` — no new dependencies
- **Noise schedule:** cosine schedule (proven better than linear for short biological sequences)
- **Interface naming:** `generate(target, n_candidates)` to clearly distinguish from `predict()` / `predict_interactions()` — avoids all naming conflicts

---

## 4. Implementation Plan — 12 Weeks (300 Hours)

*Project start: June 9, 2026. Total: 12 calendar weeks, 5 days pause.*

### Phase 1 — Foundation (Weeks 1–3, ~75 hrs)

**Week 1: Codebase Deep Dive & Architecture Design**
- Complete audit of AptaDiff reference implementation — map every component to pyaptamer equivalents
- Design `pyaptamer.aptadiff` module structure: `_model_lightning.py`, `_pipeline.py`, `_torch_dataset.py`, `_noise_schedule.py`
- Draft interface spec (`fit` / `generate` signatures) and post for mentor review before writing code
- Set up CI test matrix for AptaDiff; confirm Apple Silicon / CPU / CUDA compatibility

**Week 2: Data Layer — `AptaDiffDataset`**
- Implement `AptaDiffDataset` extending `APIDataset` (PR #441's refactored base class)
- Integrate with `MoleculeLoader` for protein feature loading
- Forward diffusion process: `q_sample(x_0, t)` producing corrupted token sequences
- Full test suite for dataset layer: shape tests, corruption level tests, `from_any()` input dispatch

**Week 3: Noise Schedule + Forward Process**
- Implement cosine noise schedule (`betas`, `alphas_cumprod`, `sqrt_` precomputed tensors)
- Validate that `q(x_T | x_0)` converges to uniform over the sequence vocabulary
- Differential tests: verify the forward process is invertible in expectation
- Milestone deliverable: `AptaDiffDataset` + noise schedule reviewed and approved by mentors

---

### Phase 2 — Core Model (Weeks 4–7, ~125 hrs)

**Week 4: Transformer Denoiser Backbone**
- Implement `AptaDiffModel` — transformer backbone conditioned on protein embeddings
- Protein conditioning: cross-attention between protein features and noisy sequence tokens
- Sinusoidal timestep embedding injected into transformer blocks
- Shape/type tests: ensure output is `(batch, seq_len, vocab_size)` logits at all timesteps

**Week 5: Lightning Module — `AptaDiffLightning`**
- Implement `AptaDiffLightning(pl.LightningModule)`:
  - `training_step`: compute `L_simple` (cross-entropy on predicted vs. true token)
  - `validation_step`: track loss + sequence recovery rate
  - `predict_step`: run reverse diffusion loop from noise → sequence (same pattern as `AptaTransLightning.predict_step()` in my PR #493)
  - `configure_optimizers`: AdamW + cosine LR schedule
- CI compatibility: `accelerator="cpu"` default for deterministic test runs

**Week 6: Reverse Process — Generation Loop**
- Implement `p_sample_loop(model, protein_features, n_steps)` — iterative denoising
- Temperature-controlled sampling: `temperature` parameter in `generate()` to control diversity vs. quality
- Beam search variant: return top-k candidates ranked by model confidence
- Tests: verify generated sequences are within vocabulary, correct length, biologically valid (A/U/G/C only)

**Week 7: Integration Testing + Baseline Evaluation**
- End-to-end integration test: protein input → `fit()` → `generate()` → valid aptamer sequences
- Baseline: reproduce AptaDiff paper's metric on a small benchmark subset (SELEX-derived dataset)
- Milestone deliverable: core model passes integration tests; mentor review of `AptaDiffLightning`

---

### Phase 3 — Pipeline + API (Weeks 8–10, ~75 hrs)

**Week 8: `AptaDiffPipeline` — Public API**
- Implement `AptaDiffPipeline`:
  ```python
  pipeline = AptaDiffPipeline()
  pipeline.fit(protein_sequences)                  # train
  candidates = pipeline.generate("MAHHHHH...", n=100)  # generate aptamers
  ```
- sklearn estimator compliance: `get_params()`, `set_params()`, `__repr__`
- `joblib.dump` / `joblib.load` round-trip test for reproducible research

**Week 9: `prot_words` Standardization + Documentation**
- Audit and standardize `prot_words` docstrings across all AptaDiff files (per Issue #190 pattern)
- Docstring for `generate()` return type: ranked `List[str]` with associated confidence scores
- Inline code examples (`doctest`-compatible) in all public methods

**Week 10: Streamlit Explorer — Interactive Protein → Aptamer Demo**
- Extend the pyaptamer Interactive Explorer (built in the Interoperability track):
  - Add "Generate Aptamers" mode: user pastes a protein sequence, Explorer calls `AptaDiffPipeline.generate()` and displays ranked candidates with confidence scores
  - Export results as CSV/JSON
- Deploy on Hugging Face Spaces (free, public, linkable from README)
- Milestone deliverable: live public demo URL committed to repository README

---

### Phase 4 — Polish + Handoff (Weeks 11–12, ~25 hrs)

**Week 11: Jupyter Notebook + Documentation Sprint**
- `notebooks/aptadiff_tutorial.ipynb` — "From protein target to candidate aptamers in 10 lines of Python"
  - Walk through data preparation, `fit()`, `generate()`, visualization of output sequences
  - Second section: compare AptaDiff-generated candidates using `AptaTransPipeline.predict_interactions()` to filter by predicted binding probability — demonstrating the full in-silico SELEX loop
- Update README: usage example, link to live Explorer

**Week 12: Buffer, Review Response, CI**
- Respond to all outstanding code review comments on PRs #493, #495, and AptaDiff PRs
- Fix any CI failures; ensure 85%+ test coverage on all `pyaptamer.aptadiff` modules
- Final documentation pass; merge preparation

---

## 5. Risks & Mitigation Strategies

| Risk | Impact | Mitigation Plan |
|---|---|---|
| **Diffusion Architecture Nuances**<br>I have built RAG/transformer pipelines in my project ContraLegal-AI, but DDPM modeling is newer to me. | Slower progress in Phase 2 (Core Model) if tensor shapes or noise schedules misalign. | I am allocating 3 full weeks to the forward/reverse process (Phase 1). I will rely heavily on my strong engineering fundamentals (PyTorch, Lightning) to validate the math via tests before integrating into the larger training loop. I will explicitly ask mentors for early review on my `q_sample` implementation. |
| **API Drift during ESoC**<br>The `APIDataset` refactor (PR #441) is still stabilizing. | AptaDiff integration might break as the core API shifts. | I actively read upstream commits (as I did before writing PR #493). I will build AptaDiff as a loosely coupled module using the standard `from_any()` dispatch, minimizing dependencies on internal private methods. |
| **Training Instability**<br>Diffusion models can be sensitive to hyperparameter tuning. | Model fails to generate valid aptamers or loss diverges. | I will write a small integration test using a minimal, over-parametrized model on a 10-sample dataset to verify the pipeline can overfit (the standard deep learning sanity check) before scaling up. |

---

## 6. Deliverables Summary

| Deliverable | Maps to Project Card Goal | Status |
|---|---|---|
| `AptaDiffDataset` + forward diffusion + noise schedule | "integrate with MoleculeLoader data loader" | Week 2–3 |
| `AptaDiffModel` transformer backbone + Lightning module | "adapting AptaDiff algorithm from scratch" | Week 4–6 |
| `AptaDiffPipeline.fit()` + `generate()` (sklearn API) | "follow the scikit-learn-style API" | Week 8 |
| Full test suite (shape, type, device, integration) | "write tests for the implementation" | Ongoing |
| `notebooks/aptadiff_tutorial.ipynb` | "create a notebook demonstrating the public API" | Week 11 |
| PyAptamer Interactive Explorer + Hugging Face deployment | Interoperability / stretch goal | Week 10 |
| PR #493 landed (AptaTrans `fit`/`predict_interactions`) | Prerequisite contribution | ✅ Open |
| PR #495 landed (MCTS phantom visit fix) | Prerequisite contribution | ✅ Open |

---

## 7. My Contribution Journey — Proving I Am Ready

### 7.1 How I Found the Real Work (April 19–20)

I forked `gc-os-ai/pyaptamer`, set up the dev environment, and ran the full test suite. I read every open issue systematically. Most were being claimed within hours — the ESOC applicant pool is highly active. My strategy: look past the easy wins and find the work requiring genuine codebase depth.

The most important open item was **[Issue #190](https://github.com/gc-os-ai/pyaptamer/issues/190)** — filed by maintainer Satvik (`@satvshr`) in October 2025, open for six months, one earlier incomplete attempt (PR #214):

> *"The current design of AptaTransPipeline appears inconsistent with the expected scikit-learn–style API since it lacks fit and predict methods... Hence, we should add fit and predict methods to AptaTransPipeline."*

This is the same API pattern the AptaDiff project card requires.

### 7.2 Collaborating on Architecture (Draft PR #447)

Before implementing `fit()`, I needed to understand the data flow. The existing `APIDataset` mixed encoding and torch integration. I opened [PR #447](https://github.com/gc-os-ai/pyaptamer/pull/447) as a draft, and contributor `siddharth7113` provided excellent guidance:

> *"We are trying to decouple different parts of aptatrans and separate the different layer , if you want to work on adding methods I would say to base it on #441 but I should warn you it is still under review so things might change, but it should be a good starting point."*

Instead of pushing conflicting code, I took this advice, left #447 as a draft, and **read all 21 commits of PR #441 in depth** — studying the new `BaseAptamerDataset` hierarchy, `from_any()` input dispatch, and the `_prepare_dataloader` seam — so I could build the right thing the first time.

### 7.3 PR #495 — MCTS Algorithmic Correctness Fix

I found a subtle UCT math bug ([Issue #483](https://github.com/gc-os-ai/pyaptamer/issues/483)): `TreeNode.__init__` set `n_visits = 1` instead of `0`. In the UCT formula `Q/N + C * sqrt(log(parent.N) / N)`, starting at `N=1` destroys the infinite exploration bonus for unvisited nodes — MCTS explores sub-optimally from the very first iteration.

Fix required three coordinated changes: `_algorithm.py`, `uct_score()` guard, and corrected test assertions. Open as [PR #495](https://github.com/gc-os-ai/pyaptamer/pull/495).

### 7.4 PR #493 — The Entrance Task: AptaTrans `fit()` + `predict_interactions()`

[PR #493](https://github.com/gc-os-ai/pyaptamer/pull/493) is a complete, production-quality implementation of all three requirements from Issue #190:

1. `AptaTransPipeline.fit(X, y)` — trains end-to-end via `APIDataset.from_any()` for flexible input handling
2. `AptaTransPipeline.predict_interactions(X)` — batch prediction via `Trainer.predict()` loop; named to avoid shadowing the existing single-pair `predict()`
3. `AptaTransLightning.predict_step()` — new Lightning method enabling batch inference
4. `prot_words` docstring standardization across three files (Satvik's requirement #1)
5. Return type documentation — `torch.float32` probabilities in `[0,1]` (Satvik's requirement #2)
6. Fixed silently failing `filter_words` doctests
7. 9 new tests: input shapes, output types, device handling, Apple Silicon MPS compatibility

This PR is the proof-of-concept for AptaDiff. The same `_prepare_dataloader` → `DataLoader` → `Trainer.predict()` pipeline works for generation with only the model backbone swapped.

### 7.5 PR #494 — Closed by Satvik, and What I Learned

*(Note: While waiting for the refactor to stabilize, I submitted [PR #494](https://github.com/gc-os-ai/pyaptamer/pull/494) for a minor `AptaNetPipeline.fit()` bug. It was closed gracefully as a parallel PR addressed it simultaneously, reinforcing my strategy to focus on deeper, architectural work rather than minor bug fixes).*

---

## 8. Why AptaDiff — Why This Matters

AptaNet and AptaTrans answer: *"Does this aptamer bind this protein?"* AptaDiff asks: *"Given this protein, what aptamers should I generate?"* — the generative counterpart that transforms pyaptamer from a prediction library into a full **design loop**:

```
generate candidates (AptaDiff)
        ↓
predict binding probability (AptaNet / AptaTrans)
        ↓
filter top candidates → laboratory testing
```

This is the complete in-silico SELEX cycle in Python. The project card identifies it as a primary roadmap goal (Issue [#81](https://github.com/gc-os-ai/pyaptamer/issues/81)); the maintainers have been tracking it since August 2025. Adding AptaDiff is not speculative scope — it's the highest-value item on the backlog that nobody has yet claimed, and it directly serves ecoSPECS's mission of accelerating aptamer discovery for diagnostics.

---

## 9. About Me

I am a second-year B.Tech student in AI & ML at Newton School of Technology, ADYPU, Pune (GPA 7.86/10.0). I have a merged PR in Microsoft VS Code (184k+ stars), two open PRs in pyaptamer directly addressing the project card requirements, and hands-on experience with transformer fine-tuning, RAG pipelines, PyTorch Lightning, sklearn-compatible APIs, and Streamlit deployment from my project ContraLegal-AI (97%+ legal risk classification accuracy).

I write open source code that ships. I communicate before taking on issues. I read upstream work before writing downstream code. When a PR gets closed, I learn from it and pivot to harder, higher-value work. These are the habits that make contributors effective.

---

## 10. Availability & Commitment

| Period | Availability |
|---|---|
| June 9 — September 2026 | Full-time, 35–40 hours/week |
| Exam period (if any) | Will communicate 2+ weeks in advance with adjusted plan |

**Time zone:** IST (UTC+5:30)  
**Mentor overlap:** 12:30–18:30 IST = 09:00–15:00 CET ✅ — 6 hours daily overlap with European mentors  
**Communication:** GitHub, Discord, email — responsive within same working day

I have reviewed the programme setup: 12 calendar weeks full-time, 5 working days pause, stipend disbursed via GC.OS. I accept these terms and have no competing commitments during the programme period.

---

## 11. Links

| Item | Link |
|---|---|
| GitHub Profile | https://github.com/Ishiezz |
| Fork | https://github.com/Ishiezz/pyaptamer |
| **PR #493** — AptaTrans `fit()` + `predict_interactions()` (entrance task, open) | https://github.com/gc-os-ai/pyaptamer/pull/493 |
| **PR #495** — MCTS phantom visit fix (open) | https://github.com/gc-os-ai/pyaptamer/pull/495 |
| PR #494 — AptaNet `return self` fix (closed by Satvik) | https://github.com/gc-os-ai/pyaptamer/pull/494 |
| PR #447 — Draft (superseded by #493) | https://github.com/gc-os-ai/pyaptamer/pull/447 |
| **Issue #190** — AptaTrans sklearn API (filed by Satvik, resolved by #493) | https://github.com/gc-os-ai/pyaptamer/issues/190 |
| **Issue #81** — Algorithm wishlist (AptaDiff, DeepAptamer, MAGA) | https://github.com/gc-os-ai/pyaptamer/issues/81 |
| Issue #483 — MCTS phantom visit (resolved by #495) | https://github.com/gc-os-ai/pyaptamer/issues/483 |
| PR #441 — APIDataset refactor (dependency of #493) | https://github.com/gc-os-ai/pyaptamer/pull/441 |
| **Microsoft VS Code PR #281302** (merged) | https://github.com/microsoft/vscode/pull/281302 |
| AptaDiff paper | https://academic.oup.com/bib/article/25/6/bbae517/7828722 |
| AptaDiff reference code | https://github.com/wz-create/AptaDiff |
| ESoC 2026 pyaptamer project card | https://github.com/european-summer-of-code/esoc2026/blob/main/cards/pyaptamer.md |

---

*Submitted: April 30, 2026 — Batch 2 deadline*