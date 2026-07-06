"""Guideline Logic Agent — IASLC 9th edition TNM rule adjudication.

Responsibilities:
- Accept T, N, M evidence from Vision and Clinical Context agents
- Apply IASLC 9th edition deterministic staging rules
- Resolve conflicts between agent outputs (conservative adjudication)
- Return final TNM stage with a structured rationale trace
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agents.vision import TFactorEvidence
from agents.clinical_context import NMFactorEvidence


class ConflictResolution(str, Enum):
    CONSERVATIVE = "conservative"   # always take the higher/worse stage
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
    """Rule-based IASLC 9th edition TNM adjudication agent.

    This agent acts as the final arbiter in the multi-agent pipeline.
    It validates T/N/M evidence from upstream agents, applies deterministic
    IASLC staging rules, resolves conflicts, and produces a structured
    StagingResult with a full rationale trace.

    Usage:
        agent = GuidelineLogicAgent(config)
        result = agent.run(t_evidence, nm_evidence, patient_id="P001")
    """

    # IASLC 9th edition stage grouping table (T × N × M → stage)
    # Populated from configs/iaslc_rules.yaml at load_rules()
    _STAGE_TABLE: dict[tuple[str, str, str], str] = {}

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.conflict_resolution = ConflictResolution(
            config.get("adjudication", {}).get(
                "conflict_resolution", ConflictResolution.CONSERVATIVE
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_rules(self) -> None:
        """Load IASLC stage grouping table from configs/iaslc_rules.yaml."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(
        self,
        t_evidence: TFactorEvidence,
        nm_evidence: NMFactorEvidence,
        patient_id: str | None = None,
    ) -> StagingResult:
        """Adjudicate T/N/M evidence and return final TNM stage.

        Args:
            t_evidence: Output from VisionAgent.
            nm_evidence: Output from ClinicalContextAgent.
            patient_id: Optional patient identifier for result metadata.

        Returns:
            StagingResult with tnm_string, overall_stage, and rationale.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pipeline steps (called internally by run())
    # ------------------------------------------------------------------

    def validate_inputs(
        self, t_evidence: TFactorEvidence, nm_evidence: NMFactorEvidence
    ) -> list[str]:
        """Check for missing or implausible values; return list of warnings."""
        raise NotImplementedError

    def resolve_conflicts(
        self,
        t_evidence: TFactorEvidence,
        nm_evidence: NMFactorEvidence,
    ) -> tuple[str, str, str]:
        """Apply conflict resolution strategy; return (T, N, M) strings."""
        raise NotImplementedError

    def apply_staging_rules(self, t: str, n: str, m: str) -> str:
        """Look up IASLC 9th edition stage group from (T, N, M) triple."""
        raise NotImplementedError

    def build_rationale(
        self,
        t: str,
        n: str,
        m: str,
        stage: str,
        conflicts: list[str],
        rules: list[str],
    ) -> StagingRationale:
        """Construct the structured rationale trace for the staging decision."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def tnm_string(t: str, n: str, m: str) -> str:
        """Format canonical TNM string, e.g. 'T2aN1M0'."""
        return f"{t}{n}{m}"

    def __repr__(self) -> str:  # noqa: D105
        edition = self.config.get("edition", "IASLC_9th_2024")
        return f"GuidelineLogicAgent(edition={edition}, resolution={self.conflict_resolution})"
