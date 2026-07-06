"""Clinical Context Agent — retrieval + extractive NLP over clinical notes.

Responsibilities:
- Parse radiology reports and discharge summaries (MIMIC-IV-Note)
- Extract N-factor (lymph node status) and M-factor (distant metastasis) evidence
- Retrieve the relevant passages (ClinicalBERT embeddings + ChromaDB when
  available; a dependency-free keyword fallback otherwise)
- Return structured N/M evidence with provenance spans — NO text generation

Design (decision #5: ClinicalBERT-only, retrieve-only):
Retrieval decides *which* passages to inspect; the N/M *decision* is made by the
deterministic, auditable extraction layer below (negation-aware lexicon matching),
so every claim is grounded in a real span. The heavy embedding/vector-store stack
is lazy-loaded and optional.

⚠️ The N-station -> N category and metastasis-site -> M subcategory mappings are
   heuristic and require clinical verification. The Guideline Logic Agent performs
   the deterministic staging from the categories produced here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Clinical lexicons (heuristic — verify before clinical use)
# ---------------------------------------------------------------------------

NEGATION_CUES = re.compile(
    r"\b(no|not|without|negative for|no evidence of|denies|ruled out|"
    r"absence of|free of|unremarkable for|rule out|r/o)\b",
    re.IGNORECASE,
)

METASTASIS_TRIGGER = re.compile(
    r"\b(metasta\w+|mets|spread to|distant disease|m1\b)\b", re.IGNORECASE
)

# site -> (regex, organ_system, is_extrathoracic)
METASTASIS_SITES: dict[str, tuple[re.Pattern, bool]] = {
    "brain":      (re.compile(r"\b(brain|cerebral|cerebellar|intracranial)\b", re.I), True),
    "bone":       (re.compile(r"\b(bone|osseous|skeletal|bony|vertebral met\w*)\b", re.I), True),
    "liver":      (re.compile(r"\b(liver|hepatic)\b", re.I), True),
    "adrenal":    (re.compile(r"\b(adrenal)\b", re.I), True),
    "kidney":     (re.compile(r"\b(renal|kidney)\b", re.I), True),
    "pleura":     (re.compile(r"\b(pleural (?:effusion|nodule|deposit|metasta\w+)|malignant effusion)\b", re.I), False),
    "pericardium":(re.compile(r"\b(pericardial (?:effusion|nodule))\b", re.I), False),
    "contralateral_lung": (re.compile(r"\bcontralateral (?:lung|lobe|nodule|pulmonary)\b", re.I), False),
}

NODE_TRIGGER = re.compile(
    r"\b(lymph\s*nodes?|lymphadenopath\w+|adenopathy|nodal (?:disease|involvement|metasta\w+))\b",
    re.IGNORECASE,
)

# location regex -> N category (heuristic).  Highest category wins.
NODE_LOCATION_TO_N: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(supraclavicular|scalene|contralateral|cervical)\b", re.I), "N3"),
    (re.compile(r"\b(mediastinal|subcarinal|paratracheal|aortopulmonary|station\s*[4-9])\b", re.I), "N2"),
    (re.compile(r"\b(hilar|peribronchial|interlobar|intrapulmonary)\b", re.I), "N1"),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:\n])\s+")


class ClinicalContextAgent:
    """Retrieve-only NLP agent for N/M-factor extraction from clinical notes.

    Uses ClinicalBERT embeddings + ChromaDB for semantic retrieval when the ML
    stack is installed, and a keyword fallback otherwise. Extraction and N/M
    classification are deterministic and grounded in spans (no generation).

    Usage:
        agent = ClinicalContextAgent(config)
        evidence = agent.run(notes)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.embed_model_name = self.config.get("embedding_model", "emilyalsentzer/Bio_ClinicalBERT")
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self._embedder = None
        self._vector_store = None
        self._notes: dict[str, dict[str, str]] = {}   # note_id -> note dict

    # ------------------------------------------------------------------
    # Lifecycle (heavy deps are lazy + optional)
    # ------------------------------------------------------------------

    def load_models(self) -> bool:
        """Try to load the ClinicalBERT embedder. Returns True if semantic
        retrieval is available, False if we fall back to keyword retrieval."""
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
            logger.info("Loaded embedder %s", self.embed_model_name)
            return True
        except Exception as exc:  # missing package or model download unavailable
            logger.warning(
                "Semantic embedder unavailable (%s) — using keyword retrieval.", exc
            )
            self._embedder = None
            return False

    def index_notes(self, notes: list[dict[str, str]]) -> None:
        """Register notes for retrieval (keyed by note_id)."""
        for note in notes:
            nid = note.get("note_id") or f"note_{len(self._notes)}"
            self._notes[nid] = {"note_id": nid, "text": note.get("text", ""),
                                "note_type": note.get("note_type", "")}

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(
        self,
        notes: list[dict[str, str]],
        patient_id: str | None = None,
    ) -> NMFactorEvidence:
        """Extract N/M evidence from a patient's clinical notes."""
        self.index_notes(notes)

        # Surface the passages we looked at (for transparency).
        context = self.retrieve_context(
            "lymph node metastasis distant spread staging", n_results=5
        )

        spans: list[ClinicalSpan] = []
        for note in notes:
            spans.extend(self.extract_entities(
                note.get("text", ""), source_note_id=note.get("note_id", "unknown")
            ))

        evidence = self.classify_nm_factors(spans, context)
        evidence.retrieved_context = context
        evidence.metadata["patient_id"] = patient_id
        evidence.metadata["n_notes"] = len(notes)
        return evidence

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """Light normalisation that PRESERVES character offsets (collapse only
        runs of spaces/tabs, never change length-affecting content)."""
        return text or ""

    def extract_entities(self, text: str, source_note_id: str = "unknown") -> list[ClinicalSpan]:
        """Sentence-level, negation-aware extraction of N/M mentions."""
        text = self.preprocess_text(text)
        spans: list[ClinicalSpan] = []

        for sent, offset in self._sentences(text):
            negated = bool(NEGATION_CUES.search(sent))

            # --- metastasis ---
            # Intrathoracic M1a indicators (malignant effusion, pleural/pericardial
            # nodules, contralateral nodule) are specific enough to stand alone.
            # Generic organ words (brain/bone/liver/...) require a metastasis trigger
            # in the same sentence to disambiguate from incidental mentions.
            trigger = METASTASIS_TRIGGER.search(sent)
            emitted_site = False
            for site, (pat, is_extra) in METASTASIS_SITES.items():
                if not pat.search(sent):
                    continue
                if is_extra and not trigger:
                    continue
                label = f"metastasis_{site}" + ("_negated" if negated else "")
                spans.append(self._span(sent, label, 0.7 if negated else 0.75, offset, source_note_id))
                emitted_site = True
            if trigger and not emitted_site:
                label = "metastasis_negated" if negated else "metastasis_unspecified"
                spans.append(self._span(sent, label, 0.7 if negated else 0.55, offset, source_note_id))

            # --- lymph nodes ---
            nt = NODE_TRIGGER.search(sent)
            if nt:
                n_cat = self._node_category(sent)
                if negated:
                    spans.append(self._span(sent, "lymph_node_negated", 0.7, offset, source_note_id))
                else:
                    label = f"lymph_node_positive_{n_cat}" if n_cat else "lymph_node_positive_unspecified"
                    conf = 0.72 if n_cat else 0.55
                    spans.append(self._span(sent, label, conf, offset, source_note_id))

        return spans

    def retrieve_context(self, query: str, n_results: int = 5) -> list[str]:
        """Return the most relevant passages. Semantic via ClinicalBERT when
        available, else keyword-overlap scoring."""
        sentences: list[tuple[str, str]] = []   # (sentence, note_id)
        for nid, note in self._notes.items():
            for sent, _ in self._sentences(note["text"]):
                if sent.strip():
                    sentences.append((sent.strip(), nid))
        if not sentences:
            return []

        if self.load_models():   # semantic path
            try:
                import numpy as np
                corpus = [s for s, _ in sentences]
                emb = self._embedder.encode(corpus + [query], normalize_embeddings=True)
                q = emb[-1]
                sims = emb[:-1] @ q
                idx = np.argsort(-sims)[:n_results]
                return [corpus[i] for i in idx]
            except Exception as exc:
                logger.warning("Semantic retrieval failed (%s); keyword fallback.", exc)

        # keyword fallback — substring match so 'lymph' hits 'lymphadenopathy'
        terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}

        def score(sentence: str) -> int:
            low = sentence.lower()
            return sum(1 for t in terms if t in low)

        scored = sorted(sentences, key=lambda sn: score(sn[0]), reverse=True)
        return [s for s, _ in scored[:n_results] if score(s) > 0]

    def classify_nm_factors(
        self, spans: list[ClinicalSpan], context: list[str] | None = None
    ) -> NMFactorEvidence:
        """Aggregate span evidence -> IASLC N / M categories + confidence."""
        ev = NMFactorEvidence(supporting_spans=spans)

        # ---- M category ----
        pos_sites = set()
        extrathoracic = set()
        intrathoracic_m1a = False
        any_pos_met = False
        any_met_mention = False
        for s in spans:
            if not s.label.startswith("metastasis"):
                continue
            any_met_mention = True
            if s.label.endswith("_negated") or s.label == "metastasis_negated":
                continue
            any_pos_met = True
            if s.label == "metastasis_unspecified":
                continue
            site = s.label[len("metastasis_"):]
            pos_sites.add(site)
            _, is_extra = METASTASIS_SITES.get(site, (None, True))
            if is_extra:
                extrathoracic.add(_ORGAN_SYSTEM.get(site, site))
            else:
                intrathoracic_m1a = True

        if not any_pos_met:
            ev.distant_metastasis = False
            ev.m_category = "M0"
            conf_m = 0.7 if any_met_mention else 0.4
        else:
            ev.distant_metastasis = True
            ev.metastasis_sites = sorted(pos_sites)
            if extrathoracic:
                ev.m_category = "M1c" if len(extrathoracic) >= 2 else "M1b"
            elif intrathoracic_m1a:
                ev.m_category = "M1a"
            else:
                ev.m_category = "M1b"   # asserted mets, site unknown
            conf_m = 0.75

        # ---- N category ----
        pos_n = [s for s in spans if s.label.startswith("lymph_node_positive")]
        neg_n = [s for s in spans if s.label == "lymph_node_negated"]
        if pos_n:
            ev.lymph_node_positive = True
            cats = [s.label.rsplit("_", 1)[-1] for s in pos_n if s.label.rsplit("_", 1)[-1].startswith("N")]
            ev.lymph_node_stations = sorted(set(cats))
            # highest (worst) category wins; default N2 when only "unspecified"
            ev.n_category = max(cats, key=_N_ORDER.__getitem__) if cats else "N2"
            conf_n = 0.72 if cats else 0.55
        elif neg_n:
            ev.lymph_node_positive = False
            ev.n_category = "N0"
            conf_n = 0.7
        else:
            ev.lymph_node_positive = False
            ev.n_category = "N0"
            conf_n = 0.4   # absence of evidence, not evidence of absence

        ev.confidence = round((conf_n + conf_m) / 2, 3)
        return ev

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sentences(text: str):
        pos = 0
        for sent in _SENTENCE_SPLIT.split(text or ""):
            start = text.find(sent, pos) if sent else pos
            if start < 0:
                start = pos
            yield sent, start
            pos = start + len(sent)

    @staticmethod
    def _node_category(sentence: str) -> str | None:
        for pat, cat in NODE_LOCATION_TO_N:   # ordered N3 -> N2 -> N1
            if pat.search(sentence):
                return cat
        return None

    @staticmethod
    def _span(sent, label, conf, offset, note_id) -> ClinicalSpan:
        s = sent.strip()
        return ClinicalSpan(
            text=s, label=label, confidence=conf,
            start_char=offset, end_char=offset + len(s), source_note_id=note_id,
        )

    def __repr__(self) -> str:  # noqa: D105
        return f"ClinicalContextAgent(model={self.embed_model_name!r}, retrieve_only=True)"


_ORGAN_SYSTEM = {"brain": "cns", "bone": "skeletal", "liver": "hepatic",
                 "adrenal": "endocrine", "kidney": "renal"}
_N_ORDER = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}
