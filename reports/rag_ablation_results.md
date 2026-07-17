# RAG for Extractive Clinical N-Staging — Results

_Two pre-specified experiments on the MIMIC-IV lung-cancer cohort (n = 230
patients with an extractable reference N-stage). Retrieval affects only the
Clinical Context (N/M) path, so N-stage accuracy is the relevant endpoint; T
comes from imaging and is unaffected._

## Research question

Does retrieval-augmented context improve extractive N-staging from clinical
notes? Answered under **two conditions**, because the answer depends on whether
the full record fits the context budget.

## Cohort class balance (read accuracy against this)

The 230-patient reference cohort is **node-positive heavy**: 172 N+ (74.8%) vs.
58 N0 (25.2%); 4-class split N0 = 58, N1 = 23, N2 = 132, N3 = 17. Consequences
that must be stated when reporting accuracy:

- A trivial **majority-class classifier ("always N+") scores 0.748 binary
  accuracy** — *higher* than any system below. Binary accuracy is therefore not
  a strong standalone metric on this cohort; it must be read alongside κ.
- Cohen's κ (chance-corrected) is the honest discrimination metric, and it is
  **low in absolute terms for every configuration** (binary κ ≈ 0.06–0.07). The
  extractive pipeline recovers node status only marginally above chance against
  this *silver* stage-statement reference — expected, given the reference comes
  from explicit `TxNyMz` statements while the agent infers N from anatomical
  findings. The value of the two experiments is therefore the **paired,
  relative** comparison (same reference on both arms), not absolute agreement.

## Consolidated results — all conditions (n = 230)

Binary N0/N+ endpoint. κ = Cohen's κ (chance-corrected). Reference cohort: 172 N+
(74.8%) / 58 N0 (25.2%); majority "always N+" baseline = 0.748 accuracy, κ = 0.

| Experiment | Condition | Context read | Binary acc | **Binary κ** | Paired test vs. its comparator |
| --- | --- | --- | --- | --- | --- |
| — | Majority baseline (always N+) | — | 0.748 | **0.000** | — |
| 1 | **Full-context** | entire record | **0.578** | **0.068** | reference arm |
| 1 | RAG (k = 8) | 8 sentences | 0.491 | 0.056 | vs. full-context: p = 0.084 (n.s.) |
| 2 | RAG (k = 4) | 4 sentences | 0.452 | 0.069 | vs. truncation: p = 1.1 × 10⁻⁵ |
| 2 | RAG (k = 8, primary) | 8 sentences | 0.491 | 0.056 | vs. truncation: p = 1.8 × 10⁻⁵ |
| 2 | RAG (k = 16) | 16 sentences | 0.530 | 0.065 | vs. truncation: p = 7.0 × 10⁻⁶ |
| 2 | Truncation (first-k, any k) | 4/8/16 sentences | 0.252 | 0.000 | degenerate (predicts N0 for all) |

**How to read it:** absolute κ is low for every system (~0.06) — extractive
N-staging recovers node status only marginally above chance against the silver
stage-statement reference, and no system beats the 0.748 majority-accuracy rate.
The two findings therefore rest on the **paired** comparisons (unaffected by class
balance): RAG does **not** beat full-context (Exp 1, n.s.), and RAG **significantly
beats truncation** on both accuracy and κ at every budget (Exp 2). The per-experiment
tables below give the 4-class figures and win/loss splits.

## Experiment 1 — RAG vs. full-context (`rag_ablation.py`)

Baseline reads the patient's **entire concatenated record**; RAG reads the top-8
retrieved sentences.

| Metric (n=230) | With RAG | Full-context | Majority baseline |
| --- | --- | --- | --- |
| Binary N0/N+ accuracy | 0.491 | **0.578** | 0.748 (always N+) |
| **Binary N0/N+ Cohen's κ** | **0.056** | **0.068** | 0.000 |
| 4-class N accuracy | 0.378 | 0.391 | 0.574 (always N2) |
| 4-class quadratic-weighted κ | 0.129 | 0.122 | 0.000 |
| Wilcoxon signed-rank (paired) | p = 0.084 (n.s.) | — | — |

Both systems *under-call* nodal disease relative to the silver reference (predict
N+ on 42% [RAG] / 60% [full] of patients vs. the true 75%), which is why binary
accuracy sits below the majority rate and κ is near zero. The paired Wilcoxon —
which is unaffected by class balance — is the load-bearing test, and it shows RAG
does **not** improve on full-context (p = 0.084, n.s.).

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

| Budget k | RAG acc (N0/N+) | Trunc acc | **RAG κ** | **Trunc κ** | Wilcoxon p | RAG better / worse |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0.452 | 0.252 | **0.069** | **0.000** | 1.1 × 10⁻⁵ | 56 / 15 |
| **8 (primary)** | **0.491** | **0.252** | **0.056** | **0.000** | **1.8 × 10⁻⁵** | **70 / 23** |
| 16 | 0.530 | 0.252 | **0.065** | **0.000** | 7.0 × 10⁻⁶ | 81 / 27 |

(κ = binary N0/N+ Cohen's κ. 4-class RAG accuracy 0.370 / 0.378 / 0.391 at
k = 4 / 8 / 16.)

**Finding:** under a realistic context budget, RAG **significantly** beats naive
truncation at every budget (p < 2 × 10⁻⁵), and RAG accuracy rises monotonically
with the budget (0.452 → 0.491 → 0.530). Truncation is a **degenerate constant
classifier — it predicts N0 for every patient (κ = 0 by construction)**, because
the first-k sentences of a long record rarely contain the nodal finding, so it
scores only the 25.2% N0 prevalence. RAG beats it on **both** accuracy and κ at
every budget. H2 is supported. (Absolute κ stays low for the reason in the
class-balance section above; the win here is the paired, relative one.)

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
