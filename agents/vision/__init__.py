"""Vision Agent — CT segmentation and radiomics feature extraction.

Responsibilities:
- Load and preprocess CT volumes (DICOM → NIfTI)
- Run nnU-Net tumour segmentation
- Extract MONAI/PyRadiomics features for T-factor estimation
- Return structured T-factor evidence with confidence scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TFactorEvidence:
    """Structured output from the Vision Agent."""

    tumour_size_mm: float | None = None          # longest diameter
    tumour_volume_mm3: float | None = None
    pleural_invasion: bool | None = None
    bronchus_involvement: bool | None = None
    mediastinal_invasion: bool | None = None
    t_category: str | None = None                # T0, T1a, T1b, T1c, T2a, ...
    confidence: float = 0.0
    radiomics_features: dict[str, float] = field(default_factory=dict)
    segmentation_path: str | None = None         # path to saved NIfTI mask
    metadata: dict[str, Any] = field(default_factory=dict)


class VisionAgent:
    """CT-based T-factor estimation agent.

    This agent wraps nnU-Net for tumour segmentation and MONAI/PyRadiomics
    for feature extraction. It returns a TFactorEvidence object consumed by
    the LangGraph orchestrator.

    Usage:
        agent = VisionAgent(config)
        evidence = agent.run(ct_volume_path)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._segmentation_model = None
        self._radiomics_extractor = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """Instantiate nnU-Net and PyRadiomics extractor.

        Called lazily on first run or explicitly during warm-up.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, ct_path: str, mask_path: str | None = None) -> TFactorEvidence:
        """Full pipeline: preprocess → segment → extract features → classify T.

        Args:
            ct_path: Path to input CT volume (NIfTI or DICOM directory).
            mask_path: Optional pre-computed segmentation mask to skip nnU-Net.

        Returns:
            TFactorEvidence with T-category, confidence, and raw features.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pipeline steps (called internally by run())
    # ------------------------------------------------------------------

    def preprocess(self, ct_path: str) -> np.ndarray:
        """Resample, clip HU, normalise CT volume per config."""
        raise NotImplementedError

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Run nnU-Net inference; return binary tumour mask."""
        raise NotImplementedError

    def extract_radiomics(
        self, volume: np.ndarray, mask: np.ndarray
    ) -> dict[str, float]:
        """Compute PyRadiomics feature vector from volume + mask."""
        raise NotImplementedError

    def classify_t_factor(
        self, radiomics: dict[str, float], mask: np.ndarray
    ) -> tuple[str, float]:
        """Map features to IASLC T category (T1a–T4) with confidence."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return f"VisionAgent(backbone={self.config.get('backbone', 'nnunet_v2')})"
