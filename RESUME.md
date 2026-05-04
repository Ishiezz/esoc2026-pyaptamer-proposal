# Isha Singh

**Email:** singhishaa.24@gmail.com | **GitHub:** [github.com/Ishiezz](https://github.com/Ishiezz) | **Location:** Pune, India (IST, UTC+5:30)

---

## Open Source Contributions

### Microsoft Visual Studio Code — PR [#281302](https://github.com/microsoft/vscode/pull/281302) ✅ Merged
*January 2026 — TypeScript, Production Codebase, 184k+ ⭐, 100M+ users*

Fixed a semver comparison bug in the extension linter that caused incorrect version diagnostics for all developers targeting older VS Code versions. Patched the comparison logic to correctly handle future major releases (e.g., v2.0.0+). Passed senior engineering review in one of the world's most heavily-reviewed open source repositories.

---

### gc-os-ai/pyaptamer — 6 PRs across API, algorithms, data pipeline, and base architecture
*April–May 2026 — Python, PyTorch Lightning, scikit-learn, pytest, ruff*

**PR [#493](https://github.com/gc-os-ai/pyaptamer/pull/493) — AptaTrans `fit()` + `predict_interactions()` (Open)**
Implemented the full sklearn-compatible training and batch inference interface for `AptaTransPipeline`, resolving a 6-month-old issue filed by the core maintainer. Before writing code, read all 21 commits of the ongoing `APIDataset` architectural refactor (PR #441) to build on the correct foundation. Delivered: `fit(X, y)` via `APIDataset.from_any()` flexible dispatch, `predict_interactions(X)` via `Trainer.predict()` batch loop, `AptaTransLightning.predict_step()`, `prot_words` docstring standardization across 3 files, and 9 new tests covering input shapes, output types, device handling, and Apple Silicon MPS compatibility.

**PR [#495](https://github.com/gc-os-ai/pyaptamer/pull/495) — MCTS Phantom Visit Fix (Closed — duplicate of #484, independently correct)**
Found a subtle UCT math bug: `TreeNode.__init__` initialized `n_visits = 1` instead of `0`, destroying the infinite exploration bonus for unvisited nodes from iteration one. Required three coordinated changes across `_algorithm.py`, `uct_score()`, and test assertions. Closed gracefully after discovering PR #484 had the identical fix — handled professionally, moved on immediately.

**PR [#617](https://github.com/gc-os-ai/pyaptamer/pull/617) — `encode_rna` Docstring + `return_type` Validation (Open)**
Independently discovered (no open issue): `encode_rna` docstring was fully copy-pasted from a protein encoder — referencing "protein sequences", "amino acid patterns", and containing the typo "trunacted". Invalid `return_type` silently returned a tensor instead of raising `ValueError`. Fixed terminology, added `Raises` section, added validation. All 20 tests pass.

**PR [#630](https://github.com/gc-os-ai/pyaptamer/pull/630) — `filter_words` Empty Dict `RuntimeWarning` Fix (Open)**
Independently discovered: `filter_words` emitted a silent `RuntimeWarning: Mean of empty slice` on empty dict input — a live production risk because it is called directly in `AptaTransPipeline._init_words()`. Added `ValueError` guard, `Raises` docstring section, and `test_filter_words_empty_dict`. All 4 tests pass.

**PR [#632](https://github.com/gc-os-ai/pyaptamer/pull/632) — `GreedyEncoder` Crash + Silent Failure Fix (Open)**
Independently discovered: `GreedyEncoder(words={})` either crashed with a confusing `ValueError: max() iterable argument is empty` (when `word_max_len=None`) or silently returned an all-zeros DataFrame (when `word_max_len` was set). Added guard in `__init__`. Created the **first-ever test suite** for `GreedyEncoder` — 7 tests covering encoding, longest-match preference, padding, truncation, unknown tokens, multiple sequences, and the empty words guard. All 7 pass.

**PR [#637](https://github.com/gc-os-ai/pyaptamer/pull/637) — `BaseTransform` Abstract Method Convention Fix (Open)**
Independently discovered: `BaseTransform._fit`, `_transform`, and `_transform_element` raised `ValueError` for unimplemented abstract methods since the class was created in November 2025. Python convention — and `skbase.BaseEstimator` which `BaseTransform` inherits from — uses `NotImplementedError`. Also added missing `__author__` and `__all__` module-level attributes consistent with every other pyaptamer module.

---

## Projects

**ContraLegal-AI** — [GitHub](https://github.com/Ishiezz)

Autonomous legal intelligence platform that converts unstructured contracts into structured, multi-class risk assessments with automated redrafting. Built on Legal-BERT + FAISS + LangChain RAG pipeline. Achieves 97%+ classification accuracy with improved high-risk recall. Stack: Python, Legal-BERT, FAISS, LangChain, PyMuPDF, Scikit-learn, Streamlit, Gemini/OpenAI/Groq APIs, GitHub Actions CI/CD.

**SkillBridge** — [GitHub](https://github.com/Ishiezz)

Production-ready full-stack and iOS platform connecting blue-collar workers with employers through a verified, role-based marketplace. Stack: Node.js, TypeScript, Express, Prisma, PostgreSQL, React, Tailwind, SwiftUI, JWT, Cloudinary, Docker, Nginx, PM2, AWS (EC2 + RDS).

**Impactify** — [GitHub](https://github.com/Ishiezz)

Full-stack data analytics platform that turns raw data into actionable insights. Stack: Python, Pandas, Matplotlib, Seaborn, Streamlit, NumPy.

---

## Technical Skills

**ML & AI:** PyTorch, PyTorch Lightning, scikit-learn, Transformers (BERT, Legal-BERT), RAG pipelines, LangChain, LangGraph, FAISS, Diffusion models (studying), NumPy, Pandas, Matplotlib, Seaborn, TensorFlow, OpenCV

**Software Engineering:** Python, TypeScript, JavaScript, SQL, Node.js, Express, React, React Native, SwiftUI, HTML/CSS, Tailwind

**Infrastructure & Tools:** Docker, Docker Compose, AWS (EC2, RDS), GitHub Actions, Nginx, PM2, Git, pytest, ruff, Prisma ORM, MongoDB, PostgreSQL, MySQL

---

## Education

**B.Tech in AI & ML** — Newton School of Technology, ADYPU, Pune (2024–2028)

**Class XII** — Gyan Bharti Residential Complex (2022–2023) — 78.3%

**Class X** — Creane Memorial High School (2020–2021) — 95.0%

---

## Leadership

**Vice President — House Council, Newton School of Technology** (Aug 2025–Present)
Planned and executed academic, cultural, and sports events; coordinated cross-functional student and faculty teams.