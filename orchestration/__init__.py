"""Multi-agent orchestration for NSCLC TNM staging.

Wires the three agents into one pipeline:

        vision (T) ----\\
  START --             +--> guideline (stage) --> END
        clinical (N/M)-/

Vision and Clinical Context are independent, so they fan out from START and join
at the Guideline Logic node. Uses LangGraph when installed; otherwise a
dependency-free sequential runner executes the same nodes. Node failures are
caught and degrade gracefully (an empty evidence object is passed on, and the
error is recorded) so the pipeline always returns a StagingResult.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from agents.vision import VisionAgent, TFactorEvidence
from agents.clinical_context import ClinicalContextAgent, NMFactorEvidence
from agents.guideline_logic import GuidelineLogicAgent, StagingResult

logger = logging.getLogger(__name__)


class StagingState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    patient_id: str
    ct_path: str
    mask_path: str | None
    notes: list[dict[str, str]]
    t_evidence: TFactorEvidence
    nm_evidence: NMFactorEvidence
    result: StagingResult
    errors: Annotated[list[str], operator.add]


class StagingOrchestrator:
    """Runs the Vision / Clinical Context / Guideline Logic agents as one pipeline.

    Usage:
        orch = StagingOrchestrator(config)
        result = orch.run(patient_id, ct_path, notes, mask_path=gtv_mask)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.vision = VisionAgent(self.config.get("vision", {}))
        self.clinical = ClinicalContextAgent(self.config.get("clinical_context", {}))
        self.guideline = GuidelineLogicAgent(self.config.get("guideline_logic", {}))
        self._graph = None

    # ------------------------------------------------------------------
    # Nodes (shared by the LangGraph and sequential paths)
    # ------------------------------------------------------------------

    def _vision_node(self, state: StagingState) -> dict[str, Any]:
        try:
            ev = self.vision.run(state["ct_path"], mask_path=state.get("mask_path"))
            return {"t_evidence": ev}
        except Exception as exc:
            logger.warning("Vision agent failed: %s", exc)
            return {"t_evidence": TFactorEvidence(), "errors": [f"vision: {exc}"]}

    def _clinical_node(self, state: StagingState) -> dict[str, Any]:
        try:
            ev = self.clinical.run(state.get("notes", []), patient_id=state.get("patient_id"))
            return {"nm_evidence": ev}
        except Exception as exc:
            logger.warning("Clinical Context agent failed: %s", exc)
            return {"nm_evidence": NMFactorEvidence(), "errors": [f"clinical: {exc}"]}

    def _guideline_node(self, state: StagingState) -> dict[str, Any]:
        t = state.get("t_evidence") or TFactorEvidence()
        nm = state.get("nm_evidence") or NMFactorEvidence()
        result = self.guideline.run(t, nm, patient_id=state.get("patient_id"))
        return {"result": result}

    # ------------------------------------------------------------------
    # Graph construction / execution
    # ------------------------------------------------------------------

    def build_graph(self):
        """Compile the LangGraph pipeline, or return None if LangGraph is absent."""
        if self._graph is not None:
            return self._graph
        try:
            from langgraph.graph import StateGraph, START, END
        except ImportError:
            logger.info("LangGraph not installed — using sequential fallback.")
            return None

        g = StateGraph(StagingState)
        g.add_node("vision", self._vision_node)
        g.add_node("clinical", self._clinical_node)
        g.add_node("guideline", self._guideline_node)
        g.add_edge(START, "vision")
        g.add_edge(START, "clinical")        # fan out (independent)
        g.add_edge("vision", "guideline")
        g.add_edge("clinical", "guideline")  # join
        g.add_edge("guideline", END)
        self._graph = g.compile()
        return self._graph

    def _run_sequential(self, state: StagingState) -> StagingState:
        """Fallback executor when LangGraph is unavailable."""
        state.setdefault("errors", [])
        for node in (self._vision_node, self._clinical_node, self._guideline_node):
            update = node(state)
            errs = update.pop("errors", [])
            if errs:
                state["errors"] = state.get("errors", []) + errs
            state.update(update)
        return state

    def run(
        self,
        patient_id: str,
        ct_path: str,
        notes: list[dict[str, str]] | None = None,
        mask_path: str | None = None,
    ) -> StagingResult:
        """Stage one patient end-to-end and return the StagingResult."""
        state: StagingState = {
            "patient_id": patient_id,
            "ct_path": ct_path,
            "mask_path": mask_path,
            "notes": notes or [],
            "errors": [],
        }
        graph = self.build_graph()
        final = graph.invoke(state) if graph is not None else self._run_sequential(state)

        result = final["result"]
        if final.get("errors"):
            result.metadata["errors"] = final["errors"]
        return result

    def __repr__(self) -> str:  # noqa: D105
        backend = "langgraph" if self.build_graph() is not None else "sequential"
        return f"StagingOrchestrator(backend={backend})"
