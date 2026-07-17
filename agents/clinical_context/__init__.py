"""Clinical Context agent.

This agent reads a patient's clinical notes (radiology reports and discharge
summaries from MIMIC-IV-Note) and pulls out two things: whether the lymph nodes
look involved (the N factor) and whether there is distant spread (the M factor).

It works in two steps. First it optionally retrieves the handful of sentences most
likely to mention nodes or metastasis, using ClinicalBERT sentence embeddings when
the ML libraries are installed, or a simple keyword match when they are not.
Then a rule-based layer reads those sentences and decides N and M. I kept the
decision rule-based on purpose (no text generation), so every category the agent
outputs can be traced back to the exact sentence it came from.

Note: the way I map a node's location to an N category, and a metastasis site to
an M subcategory, is a clinical heuristic and would need a clinician to verify it.
The actual staging is done later by the Guideline Logic agent.
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


# Word lists and patterns the extractor matches on. These are heuristics I put
# together from how radiology notes are usually phrased, not a validated ontology.

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

# This only fires on nodes that are described as abnormal. Just seeing the words
# "lymph node" is not enough, because chest CTs mention nodes even when they are
# normal ("nodes measure 8 mm", "lymph nodes are unremarkable"). So I require a
# word that actually implies disease (enlarged, bulky, metastatic, and so on).
NODE_TRIGGER = re.compile(
    r"\b("
    r"lymphadenopath\w+|adenopathy"                                    # inherently pathological
    r"|(?:enlarged|bulky|prominent|pathologic\w*|necrotic|conglomerate|"
    r"fdg[-\s]?avid|hypermetabolic|metastatic|malignant)\s+(?:\w+[-\s]+){0,3}?"
    r"(?:lymph\s*)?nodes?"                                             # "enlarged ... node"
    r"|(?:lymph\s*)?nodes?\s+(?:that\s+are\s+|which\s+are\s+|are\s+|were\s+|appear\w*\s+)?"
    r"(?:enlarged|involved|positive|metastatic|bulky|prominent|pathologic\w*|abnormal)"  # "nodes are enlarged"
    r"|nodal\s+(?:disease|involvement|metasta\w+|spread|uptake)"
    r")\b",
    re.IGNORECASE,
)

# Which N category a node location suggests. If several match, the worst one wins.
NODE_LOCATION_TO_N: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(supraclavicular|scalene|contralateral|cervical)\b", re.I), "N3"),
    (re.compile(r"\b(mediastinal|subcarinal|paratracheal|aortopulmonary|station\s*[4-9])\b", re.I), "N2"),
    (re.compile(r"\b(hilar|peribronchial|interlobar|intrapulmonary)\b", re.I), "N1"),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:\n])\s+")

# The query used to pick out node/metastasis sentences when RAG is on. I use word
# stems (metasta, adenopath) so the keyword fallback still matches the different
# endings (metastasis, metastases, metastatic). The semantic path ignores this.
DEFAULT_RAG_QUERY = (
    "lymph node nodal adenopath lymphadenopath mediastin hilar supraclavic "
    "subcarinal paratracheal metasta spread distant staging tumor lesion"
)


class ClinicalContextAgent:
    """Reads clinical notes and returns N/M evidence, without generating text.

    Retrieval (ClinicalBERT embeddings if available, keyword match otherwise) picks
    which sentences to look at; a rule layer then decides N and M from those
    sentences, keeping the reasoning traceable.

    Example:
        agent = ClinicalContextAgent(config)
        evidence = agent.run(notes)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.embed_model_name = self.config.get("embedding_model", "emilyalsentzer/Bio_ClinicalBERT")
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.rag_k = self.config.get("rag_k", 8)              # passages retrieved in RAG mode
        self.rag_query = self.config.get("rag_query", DEFAULT_RAG_QUERY)
        self.use_semantic = self.config.get("use_semantic", True)  # False -> keyword retrieval only
        self._embedder = None
        self._vector_store = None
        self._notes: dict[str, dict[str, str]] = {}   # note_id -> note dict

    def load_models(self) -> bool:
        """Try to load the ClinicalBERT embedder.

        Returns True if it loaded (so we can use semantic retrieval) and False if
        the libraries are not installed, in which case we drop back to keywords.
        The model is only loaded the first time it is needed.
        """
        if not self.use_semantic:
            return False
        if self._embedder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model_name)
            logger.info("Loaded embedder %s", self.embed_model_name)
            return True
        except Exception as exc:  # missing package or model download unavailable
            logger.warning(
                "Semantic embedder unavailable (%s); using keyword retrieval.", exc
            )
            self._embedder = None
            return False

    def index_notes(self, notes: list[dict[str, str]]) -> None:
        """Register notes for retrieval (keyed by note_id)."""
        for note in notes:
            nid = note.get("note_id") or f"note_{len(self._notes)}"
            self._notes[nid] = {"note_id": nid, "text": note.get("text", ""),
                                "note_type": note.get("note_type", "")}

    def run(
        self,
        notes: list[dict[str, str]],
        patient_id: str | None = None,
        use_rag: bool = True,
    ) -> NMFactorEvidence:
        """Extract N/M evidence from a patient's clinical notes.

        The use_rag flag is the switch my ablation study turns on and off:
        - use_rag=True  : only read the top-k retrieved sentences. This focuses on
          the relevant bits and ignores the rest of a long note (old history,
          incidental findings, things that were ruled out elsewhere).
        - use_rag=False : read the whole note, with no retrieval step.
        The two settings can give different N/M answers, which is the whole point
        of the experiment.
        """
        self._notes = {}
        self.index_notes(notes)

        if use_rag:
            context = self.retrieve_context(self.rag_query, n_results=self.rag_k)
            spans: list[ClinicalSpan] = []
            for passage in context:
                spans.extend(self.extract_entities(
                    passage, source_note_id=patient_id or "retrieved"))
        else:
            context = []
            spans = []
            for note in notes:
                spans.extend(self.extract_entities(
                    note.get("text", ""), source_note_id=note.get("note_id", "unknown")))

        evidence = self.classify_nm_factors(spans, context)
        evidence.retrieved_context = context
        evidence.metadata["patient_id"] = patient_id
        evidence.metadata["n_notes"] = len(notes)
        evidence.metadata["use_rag"] = use_rag
        return evidence

    def preprocess_text(self, text: str) -> str:
        """Return the text unchanged.

        I keep this as its own step so I have one place to add normalisation later,
        but for now it deliberately does nothing that would shift character
        positions, because the spans record where each mention was found.
        """
        return text or ""

    def extract_entities(self, text: str, source_note_id: str = "unknown") -> list[ClinicalSpan]:
        """Go sentence by sentence and pull out node / metastasis mentions.

        For each sentence I first check whether it is negated ("no evidence of..."),
        then look for metastasis and lymph-node phrases, tagging each hit with a
        label and a rough confidence.
        """
        text = self.preprocess_text(text)
        spans: list[ClinicalSpan] = []

        for sent, offset in self._sentences(text):
            negated = bool(NEGATION_CUES.search(sent))

            # Metastasis.
            # Chest-specific signs (malignant effusion, pleural/pericardial nodules,
            # a contralateral nodule) are specific enough to count on their own.
            # Plain organ words like brain/bone/liver could just be an incidental
            # mention, so I only count those if the sentence also has a metastasis
            # word in it.
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

            # Lymph nodes.
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
        """Return the n_results sentences most relevant to the query.

        If ClinicalBERT loaded, rank sentences by embedding similarity; otherwise
        rank them by how many of the query's keywords they contain.
        """
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

        # keyword fallback: substring match, so 'lymph' hits 'lymphadenopathy'
        terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 2}

        def score(sentence: str) -> int:
            low = sentence.lower()
            return sum(1 for t in terms if t in low)

        scored = sorted(sentences, key=lambda sn: score(sn[0]), reverse=True)
        return [s for s, _ in scored[:n_results] if score(s) > 0]

    def classify_nm_factors(
        self, spans: list[ClinicalSpan], context: list[str] | None = None
    ) -> NMFactorEvidence:
        """Combine all the extracted mentions into a single N and M category."""
        ev = NMFactorEvidence(supporting_spans=spans)

        # Work out M first.
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

        # Now N.
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
