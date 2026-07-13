# Pre-registration — Longitudinal (budgeted) RAG for N-staging

_Written before running the experiment. Fixed in git so the analysis cannot be
retrofitted to the result._

## Motivation

The first ablation (`rag_ablation.py`) compared **RAG (top-8 sentences)** against a
**no-RAG baseline that reads the patient's entire concatenated record** (capped at
40k chars). Result: RAG did not improve N-staging (binary N0/N+ accuracy 0.491 vs
0.578; Wilcoxon p = 0.084, non-significant). Diagnosed mechanism: when the full
record fits in context, retrieval can only *lose* recall of nodal evidence — the
baseline already sees everything.

That comparison is unfair to RAG and, more importantly, **not the setting where RAG
is meant to help.** Cohort records are long — median **19 notes/patient** (max
294) — so a real system with a bounded context window *cannot* read the whole
record and must select what to read. The scientifically interesting question is
therefore about selection under a budget, not RAG vs. unlimited context.

## Hypothesis (H2)

> Under a fixed context budget of **k = 8 sentences** — which the median patient
> record far exceeds — **relevance-based retrieval selects nodal evidence better
> than naive first-k truncation**, yielding higher agreement with the reference
> N-stage.

Both arms see the **same amount of text (k sentences)**; they differ only in
*which* k. This isolates the value of retrieval as a selection strategy.

- **RAG arm:** the k sentences most relevant to the N/M query (semantic,
  Bio_ClinicalBERT).
- **Truncation arm (baseline):** the first k sentences of the concatenated record
  (the naive "fit it in the window" strategy).

## Endpoints (fixed in advance)

- **Primary:** binary N0/N+ accuracy, RAG vs truncation, at k = 8; paired Wilcoxon
  signed-rank on per-patient ordinal error |N_pred − N_ref| (two-sided, α = 0.05).
- **Secondary:** 4-class (N0–N3) accuracy; quadratic-weighted Cohen's κ.
- **Pre-declared sensitivity analysis:** k ∈ {4, 16}. Reported in full; **k = 8 is
  the single primary endpoint** and will not be reselected post hoc.

## Reference and population

- Reference N: unchanged silver standard (`reference_n`, explicit TNM/cN/pN
  statements). Evaluated only on patients with an extractable reference N —
  identical inclusion rule to the first ablation (~230 patients).
- Reference is extracted from the **full** record (explicit stage statements),
  while predictions come from anatomical *findings*; the two signals stay distinct,
  so neither arm can trivially copy the reference.

## Decision rule (commitment)

The result is reported **honestly whatever its sign.** If RAG beats truncation, it
supports H2 (RAG helps when context is bounded). If it does not, the combined story
is a strong, mechanistic negative result: retrieval helps neither against full
context nor against truncation for this extractive task. **No tuning of k, query,
rag_k, or the extractor to chase significance.** The first ablation stands as
reported; this is an additional, distinct question, not a replacement.
