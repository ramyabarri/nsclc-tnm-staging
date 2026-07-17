"""Guideline Logic agent.

This is the last stage of the pipeline. It takes the T evidence from the Vision
agent and the N/M evidence from the Clinical Context agent, then works out the
overall stage by looking the (T, N, M) combination up in the IASLC table.

I keep the actual staging table in configs/iaslc_rules.yaml rather than hard-coding
it here, so the rules stay data (easy to check against the guideline, and I can
swap editions without touching this file). When a value is missing I fall back to
a conservative default instead of guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from agents.vision import TFactorEvidence
from agents.clinical_context import NMFactorEvidence

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "iaslc_rules.yaml"


class ConflictResolution(str, Enum):
    CONSERVATIVE = "conservative"   # when ambiguous, prefer the higher/worse stage
    MAJORITY = "majority"
    WEIGHTED = "weighted"


@dataclass
class StagingRationale:
    """Step-by-step trace of how the final stage was determined."""

    t_raw: str | None = None
    n_raw: str | None = None
    m_raw: str | None = None
    conflicts_detected: list[str] = field(default_factory=list)
    rules_applied: list[str] = field(default_factory=list)
    final_t: str | None = None
    final_n: str | None = None
    final_m: str | None = None
    overall_stage: str | None = None        # IA, IB, IIA, IIB, IIIA, IIIB, IIIC, IVA, IVB
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class StagingResult:
    """Final output of the multi-agent staging pipeline."""

    patient_id: str | None = None
    tnm_string: str | None = None           # e.g. "T2aN1M0"
    overall_stage: str | None = None        # e.g. "IIB"
    confidence: float = 0.0
    rationale: StagingRationale = field(default_factory=StagingRationale)
    t_evidence: TFactorEvidence | None = None
    nm_evidence: NMFactorEvidence | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GuidelineLogicAgent:
    """Turns T/N/M evidence into the final stage using the IASLC rules.

    It checks the inputs, fills in conservative defaults for anything missing,
    looks up the stage, and returns a StagingResult that also records how it got
    there (the rationale) so the decision can be audited.

    Example:
        agent = GuidelineLogicAgent(config)
        result = agent.run(t_evidence, nm_evidence, patient_id="P001")
    """

    # (T, N, M) -> stage, populated from the rules file (exact combos only).
    _STAGE_TABLE: dict[tuple[str, str, str], str] = {}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.conflict_resolution = ConflictResolution(
            self.config.get("adjudication", {}).get(
                "conflict_resolution", ConflictResolution.CONSERVATIVE
            )
        )
        self.min_confidence: float = self.config.get("min_confidence", 0.5)
        self.edition: str = "unloaded"
        self.stage_rules: list[dict[str, str]] = []
        self.t_categories: set[str] = set()
        self.n_categories: set[str] = set()
        self.m_categories: set[str] = set()
        self._loaded = False

    def load_rules(self) -> None:
        """Read the IASLC stage table from the YAML config into memory."""
        path = Path(self.config.get("rules_path", DEFAULT_RULES_PATH))
        if not path.exists():
            raise FileNotFoundError(f"IASLC rules file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)

        self.edition = data.get("edition", "unknown")
        self.t_categories = set(data.get("t_categories", []))
        self.n_categories = set(data.get("n_categories", []))
        self.m_categories = set(data.get("m_categories", []))
        self.stage_rules = data.get("stage_groups", [])
        if not self.stage_rules:
            raise ValueError(f"No stage_groups found in {path}")

        self._STAGE_TABLE = {
            (r["t"], r["n"], r["m"]): r["stage"]
            for r in self.stage_rules
            if "any" not in (r["t"], r["n"], r["m"])
        }
        self._loaded = True
        logger.info("Loaded %d stage rules (%s)", len(self.stage_rules), self.edition)

    def run(
        self,
        t_evidence: TFactorEvidence,
        nm_evidence: NMFactorEvidence,
        patient_id: str | None = None,
    ) -> StagingResult:
        """Take the T and N/M evidence and return the final staged result."""
        if not self._loaded:
            self.load_rules()

        warnings = self.validate_inputs(t_evidence, nm_evidence)
        t, n, m = self.resolve_conflicts(t_evidence, nm_evidence)

        # Note down anywhere we had to fall back to a default value.
        conflicts: list[str] = []
        if getattr(t_evidence, "t_category", None) in (None, ""):
            conflicts.append("T defaulted to TX (no Vision evidence)")
        if getattr(nm_evidence, "n_category", None) in (None, ""):
            conflicts.append("N defaulted to NX (no Clinical Context evidence)")
        if getattr(nm_evidence, "m_category", None) in (None, ""):
            conflicts.append("M defaulted to M0 (cM0 convention)")

        stage = self.apply_staging_rules(t, n, m)
        rules_applied = [
            f"({t}, {n}, {m}) -> stage {stage}  [{self.edition}]"
        ]
        if stage == "indeterminate":
            warnings.append(
                f"No stage grouping matched ({t}, {n}, {m}); likely an unknown or "
                "incomplete category (e.g. TX/NX). Stage left indeterminate."
            )

        confidence = min(
            _conf(t_evidence),
            _conf(nm_evidence),
        )

        rationale = self.build_rationale(t, n, m, stage, conflicts, rules_applied)
        rationale.t_raw = getattr(t_evidence, "t_category", None)
        rationale.n_raw = getattr(nm_evidence, "n_category", None)
        rationale.m_raw = getattr(nm_evidence, "m_category", None)
        rationale.warnings = warnings
        rationale.confidence = confidence

        return StagingResult(
            patient_id=patient_id,
            tnm_string=self.tnm_string(t, n, m),
            overall_stage=stage,
            confidence=confidence,
            rationale=rationale,
            t_evidence=t_evidence,
            nm_evidence=nm_evidence,
        )

    def validate_inputs(
        self, t_evidence: TFactorEvidence, nm_evidence: NMFactorEvidence
    ) -> list[str]:
        """Collect warnings about missing, unknown or low-confidence inputs."""
        if not self._loaded:
            self.load_rules()

        warnings: list[str] = []
        t = getattr(t_evidence, "t_category", None)
        n = getattr(nm_evidence, "n_category", None)
        m = getattr(nm_evidence, "m_category", None)

        if not t:
            warnings.append("T category unavailable from Vision agent (using TX).")
        elif t not in self.t_categories:
            warnings.append(f"Unrecognized T category '{t}' for {self.edition}.")
        if not n:
            warnings.append("N category unavailable from Clinical Context agent (using NX).")
        elif n not in self.n_categories:
            warnings.append(f"Unrecognized N category '{n}' for {self.edition}.")
        if not m:
            warnings.append("M category unavailable; defaulting to M0 (cM0 convention).")
        elif m not in self.m_categories:
            warnings.append(f"Unrecognized M category '{m}' for {self.edition}.")

        if _conf(t_evidence) < self.min_confidence:
            warnings.append(
                f"Low Vision confidence ({_conf(t_evidence):.2f} < {self.min_confidence})."
            )
        if _conf(nm_evidence) < self.min_confidence:
            warnings.append(
                f"Low Clinical Context confidence "
                f"({_conf(nm_evidence):.2f} < {self.min_confidence})."
            )
        return warnings

    def resolve_conflicts(
        self,
        t_evidence: TFactorEvidence,
        nm_evidence: NMFactorEvidence,
    ) -> tuple[str, str, str]:
        """Pull out (T, N, M), using safe defaults when a value is missing.

        Each factor comes from a different agent (T from Vision, N and M from
        Clinical Context), so there is no real disagreement to settle here, and the
        job is just to fill gaps sensibly. Missing T or N becomes TX or NX, and a
        missing M becomes M0, which is the usual convention when nothing points to
        distant spread.
        """
        t = getattr(t_evidence, "t_category", None) or "TX"
        n = getattr(nm_evidence, "n_category", None) or "NX"
        m = getattr(nm_evidence, "m_category", None) or "M0"
        return t, n, m

    def apply_staging_rules(self, t: str, n: str, m: str) -> str:
        """Find the stage for a (T, N, M) triple; 'any' in a rule matches anything."""
        if not self._loaded:
            self.load_rules()
        for rule in self.stage_rules:
            if (
                (rule["t"] == "any" or rule["t"] == t)
                and (rule["n"] == "any" or rule["n"] == n)
                and (rule["m"] == "any" or rule["m"] == m)
            ):
                return rule["stage"]
        return "indeterminate"

    def build_rationale(
        self,
        t: str,
        n: str,
        m: str,
        stage: str,
        conflicts: list[str],
        rules: list[str],
    ) -> StagingRationale:
        """Bundle the final T/N/M, the stage, and the trace into one object."""
        return StagingRationale(
            final_t=t,
            final_n=n,
            final_m=m,
            overall_stage=stage,
            conflicts_detected=list(conflicts),
            rules_applied=list(rules),
        )

    @staticmethod
    def tnm_string(t: str, n: str, m: str) -> str:
        """Join the parts into one string, e.g. 'T2aN1M0'."""
        return f"{t}{n}{m}"

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"GuidelineLogicAgent(edition={self.edition!r}, "
            f"resolution={self.conflict_resolution.value})"
        )


def _conf(evidence: Any) -> float:
    """Read the .confidence field, treating a missing/None value as 0.0."""
    value = getattr(evidence, "confidence", 0.0)
    return float(value) if value is not None else 0.0
