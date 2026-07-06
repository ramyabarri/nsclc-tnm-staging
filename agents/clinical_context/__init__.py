"""Clinical Context Agent — NLP extraction and RAG over clinical notes.

Responsibilities:
- Parse radiology reports and discharge summaries from MIMIC-IV-Note
- Extract N-factor clues (lymph node status) and M-factor clues (metastasis)
- Embed and retrieve relevant clinical context via ChromaDB RAG
- Return structured NM-factor evidence with provenance spans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ClinicalSpan:
    """A single extracted mention with its source provenance."""

    text: str
    label: str              # e.g. "lymph_node_positive", "metastasis_brain"
    confidence: float
    start_char: int
    end_char: int
    source_note_id: str


@dataclass
class NMFactorEvidence:
    """Structured output from the Clinical Context Agent."""

    lymph_node_positive: bool | None = None
    lymph_node_stations: list[str] = field(default_factory=list)
    distant_metastasis: bool | None = None
    metastasis_sites: list[str] = field(default_factory=list)
    n_category: str | None = None           # N0, N1, N2, N3
    m_category: str | None = None           # M0, M1a, M1b, M1c
    confidence: float = 0.0
    supporting_spans: list[ClinicalSpan] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClinicalContextAgent:
    """NLP + RAG agent for N/M-factor extraction from clinical notes.

    This agent uses ClinicalBERT for named-entity recognition over
    radiology/pathology reports, and ChromaDB for retrieval-augmented
    generation of supporting clinical context.

    Usage:
        agent = ClinicalContextAgent(config)
        evidence = agent.run(notes)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._ner_model = None
        self._embedder = None
        self._vector_store = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Load ClinicalBERT NER model and initialise ChromaDB collection."""
        raise NotImplementedError

    def index_notes(self, notes: list[dict[str, str]]) -> None:
        """Embed and upsert notes into the ChromaDB vector store.

        Args:
            notes: List of dicts with keys 'note_id', 'text', 'note_type'.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(
        self,
        notes: list[dict[str, str]],
        patient_id: str | None = None,
    ) -> NMFactorEvidence:
        """Extract N/M evidence from clinical notes for one patient.

        Args:
            notes: Radiology reports and/or discharge notes for the patient.
            patient_id: Optional ID used to scope vector store retrieval.

        Returns:
            NMFactorEvidence with N/M categories, confidence, and provenance.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pipeline steps (called internally by run())
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """Clean, de-identify (PHI removal), and normalise clinical text."""
        raise NotImplementedError

    def extract_entities(self, text: str) -> list[ClinicalSpan]:
        """Run ClinicalBERT NER; return labelled mention spans."""
        raise NotImplementedError

    def retrieve_context(self, query: str, n_results: int = 5) -> list[str]:
        """Semantic search over ChromaDB for supporting passages."""
        raise NotImplementedError

    def classify_nm_factors(
        self, spans: list[ClinicalSpan], context: list[str]
    ) -> tuple[str | None, str | None, float]:
        """Aggregate span evidence → IASLC N/M categories + confidence."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        model = self.config.get("embedding_model", "Bio_ClinicalBERT")
        return f"ClinicalContextAgent(model={model})"
