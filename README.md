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
> - **I have already built the exact architecture you need.** In PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), I implemented the sklearn-compatible `fit()` and batch inference pipeline over a PyTorch Lightning backbone for `AptaTrans` — the exact same technical pattern the project card demands for AptaDiff.
> - **I write production-quality code.** I have a merged PR in Microsoft VS Code (#281302) and five open PRs in pyaptamer — including three self-discovered bug fixes filed independently without any open issue to guide me.
> - **I understand the pyaptamer internals deeply.** I read all 21 commits of the recent `APIDataset` refactor before writing my entrance task. I independently audited `_rna.py`, `_base.py`, and `_greedy.py` and filed bugs nobody else had found (PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)). I already know how to build the AptaDiff pipeline cleanly on top of `MoleculeLoader` and the `_prepare_dataloader` seam.

---

## 1. Project Abstract

I am applying to implement the **AptaDiff algorithm** in `gc-os-ai/pyaptamer` — one of the three official 2026 full projects listed in the project card. AptaDiff is a conditional diffusion model that generates aptamer sequences targeting specific proteins, complementing the existing prediction-only algorithms (AptaNet, AptaTrans) by adding generative capability to the library.

My work will deliver a complete, production-quality `pyaptamer.aptadiff` module: a `pyaptamer`-style sklearn-compatible pipeline (`fit` + `generate`), integrated with the existing `MoleculeLoader` data pipeline, comprehensive test coverage, and a Jupyter notebook demonstrating the public API end-to-end.

I am uniquely positioned for this project. In April–May 2026, I made **five substantive contributions** to `gc-os-ai/pyaptamer`: I implemented the sklearn-compatible `fit()` and `predict_interactions()` interface for `AptaTransPipeline` (PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)) — the exact same API pattern the project card specifies for AptaDiff — and independently discovered and fixed three unreported bugs across `_rna.py`, `_base.py`, and `_greedy.py` (PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)). I understand the `pyaptamer` internals because I read 21 commits of upstream work and audited the source code line by line — not because I was guided by open issues.

---

## 2. Why I Am the Right Candidate

### 2.1 I Have Already Done the Core Pattern — And More

The AptaDiff project card requires:
> *"adapting the AptaDiff algorithm from scratch to follow the scikit-learn-style API and the existing public API conventions of pyaptamer, together with appropriate test coverage."*

This is precisely what I did for AptaTrans in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), and the pattern repeats across all my contributions:

| AptaDiff requirement | My existing work |
|---|---|
| sklearn-style `fit()` API | `AptaTransPipeline.fit(X, y)` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| batch inference via Lightning `Trainer` | `AptaTransPipeline.predict_interactions()` via `Trainer.predict()` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| `_prepare_dataloader` integration | Used and extended in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| `MoleculeLoader` / `APIDataset` integration | Studied in depth; implemented `from_any()` dispatch — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| Comprehensive test coverage | 9 tests in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); 7 new tests created for `GreedyEncoder` in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |
| Algorithmic correctness | MCTS UCT fix — PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495); `encode_rna` silent failure fix — PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) |
| Input validation + error handling | `filter_words` guard — PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630); `GreedyEncoder` guard — PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |
| Docstring quality | Fixed protein/RNA terminology copy-paste error in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617); `Raises` sections added in PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |

The work is not speculative. I have already implemented key components required for AptaDiff integration and demonstrated consistent code quality across five PRs touching five different areas of the codebase.

### 2.2 Microsoft VS Code — Production Open Source Track Record

> **PR [#281302](https://github.com/microsoft/vscode/pull/281302) merged into Microsoft VS Code (184k+ ⭐)**

In January 2026, I fixed a semver comparison bug in the extension linter — a production TypeScript codebase with an extremely high bar for correctness and review. This demonstrates I can work in large, unfamiliar codebases under peer review from a senior engineering team, which is the same skill required for contributing to `pyaptamer` under ecoSPECS and GC.OS mentorship.

### 2.3 Deep Learning Stack — Direct Match to Project Requirements

The project card lists: *Python, familiarity with deep learning, PyTorch, Lightning, and scikit-learn.*

| Required skill | My evidence |
|---|---|
| Python (advanced) | PRs [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632); my project ContraLegal-AI (97%+ accuracy ML system) |
| PyTorch | `AptaTransPipeline` training loop via `Trainer` — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); MCTS fix — PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495) |
| PyTorch Lightning | `AptaTransLightning.predict_step()` in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); Apple Silicon MPS fix |
| scikit-learn interface | `fit()`, `predict_interactions()`, `score()` pattern — PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) |
| Input validation patterns | `ValueError` guards in PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — consistent with codebase conventions |
| Diffusion models | my project ContraLegal-AI uses BERT + RAG; currently studying DDPM architecture |
| Testing (pytest) | 9 tests in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493); corrected MCTS assertions in PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495); 20 passing tests after PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617); 4 passing tests after PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630); first-ever 7-test suite for `GreedyEncoder` in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) |

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

The `_prepare_dataloader` seam I used in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) (introduced by PR #441's `APIDataset` refactor) is the exact interface needed to feed protein embeddings into AptaDiff training. The `prot_words` vocabulary encoding I validated in PRs [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) and [#630](https://github.com/gc-os-ai/pyaptamer/pull/630) is the same encoding AptaDiff will use for protein conditioning. The `GreedyEncoder` I debugged in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) and `encode_rna` I hardened in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) are the tokenizers that will preprocess sequences fed into the diffusion model.

### 3.3 Implementation Decisions

- **Sequence representation:** discrete token indices (A, U, G, C + padding) — compatible with `pyaptamer`'s existing `prot_words` vocabulary encoding, validated in PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)
- **Tokenization:** will reuse `GreedyEncoder` (hardened in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)) and `encode_rna` (hardened in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617)) — both already debugged and production-ready
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
- Integrate with `MoleculeLoader` for protein feature loading; reuse `encode_rna` (hardened in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617)) and `GreedyEncoder` (hardened in PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)) for sequence tokenization
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
- Protein conditioning: cross-attention between protein features and noisy sequence tokens; reuse `filter_words` vocabulary pipeline (hardened in PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)) for protein word encoding
- Sinusoidal timestep embedding injected into transformer blocks
- Shape/type tests: ensure output is `(batch, seq_len, vocab_size)` logits at all timesteps

**Week 5: Lightning Module — `AptaDiffLightning`**
- Implement `AptaDiffLightning(pl.LightningModule)`:
  - `training_step`: compute `L_simple` (cross-entropy on predicted vs. true token)
  - `validation_step`: track loss + sequence recovery rate
  - `predict_step`: run reverse diffusion loop from noise → sequence (same pattern as `AptaTransLightning.predict_step()` in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493))
  - `configure_optimizers`: AdamW + cosine LR schedule
- CI compatibility: `accelerator="cpu"` default for deterministic test runs

**Week 6: Reverse Process — Generation Loop**
- Implement `p_sample_loop(model, protein_features, n_steps)` — iterative denoising
- Temperature-controlled sampling: `temperature` parameter in `generate()` to control diversity vs. quality
- Beam search variant: return top-k candidates ranked by model confidence
- Tests: verify generated sequences are within vocabulary, correct length, biologically valid (A/U/G/C only) — reuse `dna2rna` / `rna2vec` utilities from `_rna.py` (audited in PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617))

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
- Full input validation with clear `ValueError` messages — consistent with the guard pattern established in PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)
- sklearn estimator compliance: `get_params()`, `set_params()`, `__repr__`
- `joblib.dump` / `joblib.load` round-trip test for reproducible research

**Week 9: `prot_words` Standardization + Documentation**
- Audit and standardize `prot_words` docstrings across all AptaDiff files (per Issue #190 pattern, already applied in PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493))
- Apply consistent docstring quality standards established in PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632): correct terminology, `Raises` sections, typed `Parameters`/`Returns`
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
- Respond to all outstanding code review comments on PRs [#493](https://github.com/gc-os-ai/pyaptamer/pull/493), [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632), and AptaDiff PRs
- Fix any CI failures; ensure 85%+ test coverage on all `pyaptamer.aptadiff` modules
- Final documentation pass; merge preparation

---

## 5. Risks & Mitigation Strategies

| Risk | Impact | Mitigation Plan |
|---|---|---|
| **Diffusion Architecture Nuances**<br>I have built RAG/transformer pipelines in my project ContraLegal-AI, but DDPM modeling is newer to me. | Slower progress in Phase 2 (Core Model) if tensor shapes or noise schedules misalign. | I am allocating 3 full weeks to the forward/reverse process (Phase 1). I will validate the math via tests before integrating into the larger training loop, and ask mentors for early review on my `q_sample` implementation. |
| **API Drift during ESoC**<br>The `APIDataset` refactor (PR #441) is still stabilizing. | AptaDiff integration might break as the core API shifts. | I actively read upstream commits (as I did before writing PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)). I will build AptaDiff as a loosely coupled module using the standard `from_any()` dispatch, minimizing dependencies on internal private methods. |
| **Training Instability**<br>Diffusion models can be sensitive to hyperparameter tuning. | Model fails to generate valid aptamers or loss diverges. | I will write a small integration test using a minimal, over-parametrized model on a 10-sample dataset to verify the pipeline can overfit before scaling up. |
| **Silent failures in data pipeline**<br>Empty or malformed inputs can propagate silently. | Bad data enters training without any error signal. | I have already established the fix pattern across PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — every AptaDiff input boundary will have explicit `ValueError` guards from day one. |

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
| **PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493)** — AptaTrans `fit()` + `predict_interactions()` | Prerequisite — sklearn API pattern | ✅ Open |
| **PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495)** — MCTS phantom visit fix | Prerequisite — algorithmic correctness | ✅ Closed (duplicate of #484 — independently correct fix) |
| **PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617)** — `encode_rna` docstring fix + `return_type` validation | Bug fix — RNA tokenizer used by AptaDiff | ✅ Open |
| **PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)** — `filter_words` empty dict `RuntimeWarning` fix | Bug fix — protein vocabulary pipeline used by AptaDiff | ✅ Open |
| **PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)** — `GreedyEncoder` empty `words` crash fix + test suite | Bug fix — sequence encoder used by AptaDiff | ✅ Open |

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

### 7.3 PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495) — MCTS Algorithmic Correctness Fix

I found a subtle UCT math bug ([Issue #483](https://github.com/gc-os-ai/pyaptamer/issues/483)): `TreeNode.__init__` set `n_visits = 1` instead of `0`. In the UCT formula `Q/N + C * sqrt(log(parent.N) / N)`, starting at `N=1` destroys the infinite exploration bonus for unvisited nodes — MCTS explores sub-optimally from the very first iteration.

Fix required three coordinated changes: `_algorithm.py`, `uct_score()` guard, and corrected test assertions. Closed as duplicate of PR #484 — I had independently arrived at the identical correct fix before knowing #484 existed. Handling this gracefully — acknowledging the duplicate, commenting professionally, and moving on to find new bugs — is itself evidence of good contributor behaviour.

### 7.4 PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) — The Entrance Task: AptaTrans `fit()` + `predict_interactions()`

[PR #493](https://github.com/gc-os-ai/pyaptamer/pull/493) is a complete, production-quality implementation of all three requirements from Issue #190:

1. `AptaTransPipeline.fit(X, y)` — trains end-to-end via `APIDataset.from_any()` for flexible input handling
2. `AptaTransPipeline.predict_interactions(X)` — batch prediction via `Trainer.predict()` loop; named to avoid shadowing the existing single-pair `predict()`
3. `AptaTransLightning.predict_step()` — new Lightning method enabling batch inference
4. `prot_words` docstring standardization across three files (Satvik's requirement #1)
5. Return type documentation — `torch.float32` probabilities in `[0,1]` (Satvik's requirement #2)
6. Fixed silently failing `filter_words` doctests
7. 9 new tests: input shapes, output types, device handling, Apple Silicon MPS compatibility

This PR is the proof-of-concept for AptaDiff. The same `_prepare_dataloader` → `DataLoader` → `Trainer.predict()` pipeline works for generation with only the model backbone swapped.

### 7.5 PR [#494](https://github.com/gc-os-ai/pyaptamer/pull/494) — Closed by Satvik, and What I Learned

*(Note: While waiting for the refactor to stabilize, I submitted [PR #494](https://github.com/gc-os-ai/pyaptamer/pull/494) for a minor `AptaNetPipeline.fit()` bug. It was closed gracefully as a parallel PR addressed it simultaneously, reinforcing my strategy to focus on deeper, architectural work rather than minor bug fixes.)*

### 7.6 PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — Independent Codebase Audit (May 2026)

After closing PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495), I continued reading the codebase independently — without being guided by open issues — and found three unreported bugs by reading source code and writing reproduction scripts. All three affect components that AptaDiff will directly depend on.

**PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) — `encode_rna` docstring terminology + `return_type` validation (Issue #616)**

The `encode_rna` function in `pyaptamer/utils/_rna.py` — the primary RNA tokenizer that AptaDiff will use for sequence encoding — had its entire docstring copy-pasted from a protein encoder: incorrectly referencing "protein sequences", "amino acid patterns", and containing the typo "trunacted". More critically, passing an invalid `return_type` silently returned a tensor instead of raising `ValueError`, masking user mistakes with no feedback:

```python
encode_rna("ACG", {"A": 1}, max_len=3, return_type="invalid")
# Before: silently returned a tensor — no error raised
# After:  ValueError: `return_type` must be either 'tensor' or 'numpy', got 'invalid'.
```

Fix is consistent with how `rna2vec` handles invalid `sequence_type`. Added `test_encode_rna_invalid_return_type`. All 20 tests pass.

**PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630) — `filter_words` silent `RuntimeWarning` on empty input**

`filter_words` in `pyaptamer/utils/_base.py` — the vocabulary filtering function called directly in `AptaTransPipeline._init_words()`, and which AptaDiff's protein conditioning pipeline will also use — emitted a silent `RuntimeWarning` (Mean of empty slice) when called with an empty dict. An empty `prot_words` dict would silently propagate garbage data downstream:

```python
filter_words({})
# Before: RuntimeWarning: Mean of empty slice — silently returns {}
# After:  ValueError: `words` must not be empty.
```

Added guard, `Raises` docstring section, and `test_filter_words_empty_dict`. All 4 tests pass.

**PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — `GreedyEncoder` crash and silent failure on empty `words` dict**

`GreedyEncoder` in `pyaptamer/trafos/encode/_greedy.py` — the sequence tokenizer that feeds into both the existing AptaTrans pipeline and the planned AptaDiff pipeline — had two failure modes on `words={}`:

```python
# Mode 1 — confusing crash (default word_max_len=None):
GreedyEncoder(words={}).fit_transform(X)
# ValueError: max() iterable argument is empty  ← no indication words={} is wrong

# Mode 2 — silent all-zeros output (when word_max_len is set):
GreedyEncoder(words={}, word_max_len=3).fit_transform(X)
# Returns all-zero DataFrame silently — no error, no warning
```

Added empty guard in `__init__`. Also created the **first-ever test suite** for `GreedyEncoder` (`pyaptamer/trafos/encode/tests/test_greedy_encoder.py`) — 7 tests covering basic encoding, longest-match preference, unknown tokens, padding, truncation, multiple sequences, and the empty words guard. All 7 pass.

These three PRs demonstrate that my codebase engagement goes beyond claiming open issues — I read code to find problems that haven't been reported yet. The components hardened in PRs [#617](https://github.com/gc-os-ai/pyaptamer/pull/617), [#630](https://github.com/gc-os-ai/pyaptamer/pull/630), and [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) are the exact building blocks AptaDiff will depend on.

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

My existing contributions have already hardened the data pipeline layer that AptaDiff depends on: `encode_rna` (PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617)), `filter_words` (PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630)), and `GreedyEncoder` (PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632)) are all more robust than when I found them. The AptaDiff implementation starts on a cleaner foundation because of this preparatory work.

---

## 9. About Me

I am a second-year B.Tech student in AI & ML at Newton School of Technology, ADYPU, Pune (GPA 7.86/10.0). I have a merged PR in Microsoft VS Code (184k+ stars), and five substantive contributions to `gc-os-ai/pyaptamer`: PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) (sklearn API — the project card's core pattern), PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) (RNA tokenizer hardening), PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630) (vocabulary pipeline hardening), PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) (sequence encoder hardening + first-ever test suite). I have hands-on experience with transformer fine-tuning, RAG pipelines, PyTorch Lightning, sklearn-compatible APIs, and Streamlit deployment from my project ContraLegal-AI (97%+ legal risk classification accuracy).

I write open source code that ships. I communicate before taking on issues. I read upstream work before writing downstream code. I find bugs that haven't been reported yet. When a PR gets closed, I learn from it and pivot to harder, higher-value work. These are the habits that make contributors effective — and they are on full display across my contribution history.

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
| **PR #495** — MCTS phantom visit fix (closed — duplicate of #484, independently correct) | https://github.com/gc-os-ai/pyaptamer/pull/495 |
| **PR #617** — `encode_rna` docstring fix + `return_type` validation (Issue #616) | https://github.com/gc-os-ai/pyaptamer/pull/617 |
| **PR #630** — `filter_words` empty dict `RuntimeWarning` fix | https://github.com/gc-os-ai/pyaptamer/pull/630 |
| **PR #632** — `GreedyEncoder` empty `words` crash fix + first-ever test suite | https://github.com/gc-os-ai/pyaptamer/pull/632 |
| PR #494 — AptaNet `return self` fix (closed by Satvik) | https://github.com/gc-os-ai/pyaptamer/pull/494 |
| PR #447 — Draft (superseded by #493) | https://github.com/gc-os-ai/pyaptamer/pull/447 |
| **Issue #190** — AptaTrans sklearn API (filed by Satvik, resolved by #493) | https://github.com/gc-os-ai/pyaptamer/issues/190 |
| **Issue #81** — Algorithm wishlist (AptaDiff, DeepAptamer, MAGA) | https://github.com/gc-os-ai/pyaptamer/issues/81 |
| Issue #483 — MCTS phantom visit (resolved by #495 / #484) | https://github.com/gc-os-ai/pyaptamer/issues/483 |
| Issue #616 — `encode_rna` docstring + `return_type` bug (resolved by PR #617) | https://github.com/gc-os-ai/pyaptamer/issues/616 |
| PR #441 — APIDataset refactor (dependency of #493) | https://github.com/gc-os-ai/pyaptamer/pull/441 |
| **Microsoft VS Code PR #281302** (merged) | https://github.com/microsoft/vscode/pull/281302 |
| AptaDiff paper | https://academic.oup.com/bib/article/25/6/bbae517/7828722 |
| AptaDiff reference code | https://github.com/wz-create/AptaDiff |
| ESoC 2026 pyaptamer project card | https://github.com/european-summer-of-code/esoc2026/blob/main/cards/pyaptamer.md |

---

*Submitted: April 30, 2026 — Batch 2 deadline*
*Updated: May 2, 2026 — PRs #617, #630, #632 added*