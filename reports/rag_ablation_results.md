# RAG for Extractive Clinical N-Staging — Results

_Two pre-specified experiments on the MIMIC-IV lung-cancer cohort (n = 230
patients with an extractable reference N-stage). Retrieval affects only the
Clinical Context (N/M) path, so N-stage accuracy is the relevant endpoint; T
comes from imaging and is unaffected._

## Research question

Does retrieval-augmented context improve extractive N-staging from clinical
notes? Answered under **two conditions**, because the answer depends on whether
the full record fits the context budget.

## Experiment 1 — RAG vs. full-context (`rag_ablation.py`)

Baseline reads the patient's **entire concatenated record**; RAG reads the top-8
retrieved sentences.

| Metric (n=230) | With RAG | Full-context |
| --- | --- | --- |
| Binary N0/N+ accuracy | 0.491 | **0.578** |
| 4-class N accuracy | 0.378 | 0.391 |
| Quadratic κ | 0.129 | 0.122 |
| Wilcoxon signed-rank (paired) | p = 0.084 (n.s.) | — |

**Finding:** RAG does **not** beat reading the whole record; the trend is toward
worse, but not significant (p = 0.084). Mechanism: when the record fits in
context, retrieval can only *lose* recall of nodal evidence the baseline already
sees. This is an honest negative result for the "RAG-beats-full-context" framing.

## Experiment 2 — RAG vs. truncation under a context budget (`rag_longitudinal.py`)

Pre-registered (`reports/PREREGISTRATION_rag_longitudinal.md`, committed before
running). Cohort records are long — **median 19 notes/patient (max 294)** — so a
bounded-context system cannot read everything and must select. Both arms get the
**same budget of k sentences**; they differ only in selection: RAG takes the k
most relevant, truncation takes the first k. Primary endpoint **k = 8**; k ∈ {4,
16} pre-declared as sensitivity.

| Budget k | RAG (binary N0/N+) | Truncation | RAG 4-class | Wilcoxon p | RAG better / worse |
| --- | --- | --- | --- | --- | --- |
| 4 | 0.452 | 0.252 | 0.370 | 1.1 × 10⁻⁵ | 56 / 15 |
| **8 (primary)** | **0.491** | **0.252** | **0.378** | **1.8 × 10⁻⁵** | **70 / 23** |
| 16 | 0.530 | 0.252 | 0.391 | 7.0 × 10⁻⁶ | 81 / 27 |

**Finding:** under a realistic context budget, RAG **significantly** beats naive
truncation at every budget (p < 2 × 10⁻⁵), and RAG accuracy rises monotonically
with the budget (0.452 → 0.491 → 0.530). Truncation is flat at 0.252 with κ = 0 —
it collapses to predicting N0, because the first-k sentences of a long record
rarely contain the nodal finding. H2 is supported.

## The complete, honest picture (three-way)

```
full-context   0.578   ← read everything (Exp 1 baseline; upper bound)
RAG k=16       0.530   ← 92% of the ceiling, reading ~16 sentences
RAG k=8        0.491   ← 85% of the ceiling  (primary)
RAG k=4        0.452   ← 78% of the ceiling
truncation     0.252   ← naive budget baseline (any k), no better than guessing N0
```

Retrieval is **not** a way to beat full-context reading — when the whole record
fits, read it. But when it does **not** fit (the real deployment constraint), RAG
is a strong approximation of full-context reading: it recovers **78–92%** of the
full-context accuracy while reading only 4–16 sentences, and it beats the naive
truncation alternative by a wide, statistically significant margin. As the budget
grows, RAG degrades gracefully toward the full-context ceiling.

## Contribution framing (for the dissertation)

The novelty is **not** "RAG improves clinical staging" (Experiment 1 shows it does
not, versus full context). It is the **empirical characterisation of when
retrieval helps**: a negative result against unlimited context and a significant
positive result against bounded context, with a graceful budget–accuracy curve
and a clear mechanism. Both experiments are reported; neither replaces the other.

## Honest caveats

- **Silver reference.** N-reference is extracted from explicit staging statements,
  not adjudicated by a clinician; but the comparison is paired against the same
  reference, so the Wilcoxon test is valid regardless.
- **Truncation is the *naive* budget baseline.** First-k is the conventional
  truncation baseline in the RAG literature, but it is weak; the more demanding
  comparison (RAG vs full-context, Exp 1) is also reported and RAG loses it. We do
  not overstate: RAG beats *truncation*, not *reading everything*.
- **N-stage only.** M-staging is out of scope (see the staging evaluation report);
  RAG touches only the Clinical Context path.
- **No post-hoc selection.** k = 8 was the pre-registered primary; {4, 16} are
  reported in full as declared. No parameter was tuned to chase significance.

## Reproduce

```bash
python -m scripts.evaluation.rag_ablation                     # Exp 1 (RAG vs full-context)
python -m scripts.evaluation.rag_longitudinal --budgets 8 4 16 # Exp 2 (budgeted; k=8 primary)
```

Outputs in `results/ablation/` (gitignored — contains MIMIC subject_ids).
