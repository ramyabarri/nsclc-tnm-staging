# TotalSegmentator task-list review — is there a lesion/nodule model?

**Deliverable for mentor (Week 5, item 4).** Question: before locking in
TotalSegmentator's role in the Vision Agent, does it ship a lesion- or
nodule-specific model, or only anatomical (lung-lobe) segmentation?

## Method

Enumerated the authoritative task list from the installed package
(`totalsegmentator` v2.14.0, `nsclc-mas` env) via `map_to_binary.class_map`, and
scanned every task's class names for `nodule|lesion|tumour|cancer|metasta`.

## Finding — yes, a lung lesion model exists

TotalSegmentator v2.14.0 exposes ~50 tasks. The lesion-bearing ones are:

| Task | Classes | Relevant here? |
|---|---|---|
| `lung_nodules` | `{1: lung, 2: lung_nodules}` | **Yes** — the only lung lesion model |
| `liver_lesions` | `{1: liver_lesions}` | No (wrong organ) |
| `liver_vessels` | vessels + tumour class | No (wrong organ) |

So the near-zero DSC the mentor reviewed was **not** the best TotalSegmentator can do
— that run used the general `total`/lung-lobe model, which segments parenchyma, not
tumour. The lesion-specific `lung_nodules` task is the honest comparison target.

## Code status

- `agents/vision/inference.py` was already switched from lung-lobe to
  `task="lung_nodules"` (extracting class 2) on Jun 24 — **after** the Jun 17 baseline
  the mentor saw. The critique was correct about that run; the code has since moved on.
- Fixed a latent bug while validating: `lung_nodules` has no "fast" variant and
  raises `ValueError` on `fast=True`. Switched to `fast=False` (full-resolution model,
  ~765 MB one-time download, CPU).

## Caveat — nodule detection ≠ GTV delineation

`lung_nodules` is trained to *detect pulmonary nodules*, not to delineate the gross
tumour volume (GTV) that our RTSTRUCT-derived ground truth encodes. NSCLC-Radiomics
GTVs are frequently bulky masses, not discrete small nodules, so we should expect
`lung_nodules` to **under-segment** them. A low DSC here is a target-mismatch, not a
tuning failure.

## Recommendation

Re-run the sanity baseline with `lung_nodules` (in progress) and read the DSC:

- If DSC is materially non-zero, keep TotalSegmentator as a **comparison segmenter**
  alongside the own-cohort nnU-Net, with the nodule-vs-GTV caveat stated.
- If DSC stays near zero (likely, given the target mismatch), adopt the mentor's
  **anatomical-context** framing: use TotalSegmentator to narrow the search region to
  lung tissue / candidate nodules, and let the own-cohort nnU-Net do the GTV
  delineation. This is the more honest architecture write-up.

_Sanity-sample numbers to be appended once the Week 5 baseline run completes._
