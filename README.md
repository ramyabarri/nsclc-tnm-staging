# Multi-Agent System for Automated NSCLC TNM Staging

## Project Overview

This project develops a multi-agent AI system for automated staging of Non-Small Cell Lung Cancer (NSCLC) using the TNM classification framework. Three specialised agents — Vision, Clinical Context, and Guideline Logic — are orchestrated via LangGraph to integrate CT imaging findings, clinical notes, and evidence-based staging guidelines into a unified, explainable staging output.

---

## Research Questions

| # | Research Question |
|---|-------------------|
| RQ1 | Can a multi-agent LLM system achieve clinician-level accuracy in automated NSCLC TNM staging compared to expert consensus labels? |
| RQ2 | How does integrating radiological imaging features (Vision Agent) with unstructured clinical text (Clinical Context Agent) affect staging accuracy versus unimodal baselines? |
| RQ3 | Does incorporating structured IASLC guideline logic (Guideline Logic Agent) improve consistency and reduce staging errors relative to a prompt-only approach? |
| RQ4 | What are the failure modes of the system, and how do agent disagreements correlate with staging uncertainty and downstream clinical impact? |

---

## Datasets

| Dataset | Source | Modality | Size | Use |
|---------|--------|----------|------|-----|
| NSCLC-Radiomics | TCIA | CT (DICOM) | 422 patients | Tumour segmentation, T-factor features |
| NSCLC-Radiogenomics | TCIA | CT + genomics | 211 patients | Multimodal fusion, staging labels |
| MIMIC-IV v3.1 | PhysioNet | Structured EHR | ~300k admissions | N/M-factor features, demographics |
| MIMIC-IV-Note v2.2 | PhysioNet | Clinical notes (NLP) | ~300k notes | Radiology/pathology report extraction |

> **Data access**: TCIA datasets are publicly available. MIMIC datasets require PhysioNet credentialing and data use agreement (DUA). Raw data must **never** be committed to this repository.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  LangGraph Orchestrator              │
│                                                     │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────┐ │
│  │ Vision      │  │ Clinical Context │  │Guideline│ │
│  │ Agent       │  │ Agent            │  │Logic    │ │
│  │ (MONAI/     │  │ (ClinicalBERT/   │  │Agent    │ │
│  │  nnU-Net)   │  │  ChromaDB RAG)   │  │(IASLC)  │ │
│  └──────┬──────┘  └────────┬─────────┘  └───┬────┘ │
│         └─────────────────┬┘                │      │
│                      ┌────▼─────┐           │      │
│                      │ Staging  ◄───────────┘      │
│                      │ Reasoner │                   │
│                      └──────────┘                   │
└─────────────────────────────────────────────────────┘
```

**Vision Agent**: Processes CT volumes using nnU-Net segmentation and MONAI feature extraction to estimate T-factor (tumour size, invasion).

**Clinical Context Agent**: Parses radiology reports and clinical notes via ClinicalBERT; uses ChromaDB vector store for RAG-based evidence retrieval for N/M-factor estimation.

**Guideline Logic Agent**: Applies IASLC 9th edition TNM rules as structured logic; validates and adjudicates between agent outputs.

---

## Repository Structure

```
.
├── agents/
│   ├── vision/             # CT segmentation & radiomics agent
│   ├── clinical_context/   # NLP + RAG agent for clinical notes
│   └── guideline_logic/    # IASLC rule-based adjudication agent
├── orchestration/          # LangGraph graph definition & state management
├── data/
│   ├── raw/                # NEVER committed — see .gitignore
│   ├── processed/          # Derived features (also gitignored)
│   └── sample/             # Anonymised toy examples for unit tests
├── notebooks/
│   ├── eda/                # Exploratory data analysis
│   └── experiments/        # Ablation studies & prototyping
├── configs/                # Hydra/YAML configuration files
├── scripts/
│   ├── preprocessing/      # DICOM → NIfTI, note parsing pipelines
│   ├── training/           # Fine-tuning & segmentation training
│   └── evaluation/         # Staging evaluation harness
├── results/
│   ├── segmentation/       # nnU-Net outputs
│   ├── staging/            # Final TNM predictions
│   └── figures/            # Plots for reports
├── tests/                  # Unit & integration tests
├── docs/                   # Architecture diagrams, data dictionaries
├── reports/
│   └── weekly/             # Weekly progress reports
├── environment.yml
├── requirements.txt
└── configs/base.yaml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate nsclc-staging
```

---

## Evaluation Metrics

- TNM stage agreement (exact match, off-by-one)
- Per-component accuracy: T, N, M individually
- Cohen's Kappa vs. expert consensus
- AUC for binary early/late stage classification
- Calibration (ECE) for confidence estimates
