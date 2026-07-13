"""Vision Agent — CT segmentation and T-factor estimation.

Responsibilities:
- Load a (pre-processed) CT volume and a tumour mask (provided, or produced by
  the locally-trained nnU-Net via agents.vision.inference)
- Extract shape features (max 3D diameter, volume); full PyRadiomics vector is an
  optional lazy add-on
- Map tumour size -> IASLC T category and return structured T-factor evidence

Notes / limitations:
- The size -> T mapping is reliable from segmentation. Invasion-based upstaging
  (visceral pleura / chest wall / mediastinum / main bronchus) cannot be read
  from a binary tumour mask alone; those are accepted as OPTIONAL inputs and are
  a documented area for future work (e.g. via TotalSegmentator anatomical context).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# (z, y, x) voxel spacing in mm — matches the preprocessing pipeline.
DEFAULT_SPACING_ZYX = (3.0, 1.0, 1.0)

# IASLC T size thresholds (greatest dimension, mm) -> category. Upper-bound, mm.
T_SIZE_THRESHOLDS: list[tuple[float, str]] = [
    (10.0, "T1a"), (20.0, "T1b"), (30.0, "T1c"),
    (40.0, "T2a"), (50.0, "T2b"), (70.0, "T3"),
]  # > 70 mm -> T4
_T_BOUNDS = [10.0, 20.0, 30.0, 40.0, 50.0, 70.0]
_T_ORDER = ["Tis", "T1a", "T1b", "T1c", "T2a", "T2b", "T3", "T4"]

DEFAULT_NNUNET_MODEL_FOLDER = (
    "results/nnunet/Dataset001_NSCLCRadiomics"
    "/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
)


@dataclass
class TFactorEvidence:
    """Structured output from the Vision Agent."""

    tumour_size_mm: float | None = None          # greatest dimension
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

    Usage:
        agent = VisionAgent(config)
        evidence = agent.run(ct_path, mask_path=gtv_mask_path)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.spacing = tuple(self.config.get("spacing_zyx", DEFAULT_SPACING_ZYX))
        self.model_folder = Path(self.config.get("nnunet_model", DEFAULT_NNUNET_MODEL_FOLDER))
        self.min_tumour_voxels = int(self.config.get("min_tumour_voxels", 10))
        # Clinical T is the greatest dimension of the PRIMARY tumour, measured on
        # the axial plane. Measuring the max-3D-diameter over the whole GTV
        # over-stages: (a) the cranio-caudal diagonal inflates the number, and
        # (b) disconnected nodal/satellite deposits in the GTV get included.
        # Defaults below fix both; set diameter_method="diameter3d" and
        # use_largest_component=False to recover the naive whole-GTV behaviour.
        self.diameter_method = self.config.get("diameter_method", "axial")  # "axial" | "diameter3d"
        self.use_largest_component = bool(self.config.get("use_largest_component", True))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> None:
        """No-op: the nnU-Net predictor is loaded lazily inside segment()."""
        return None

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def run(self, ct_path: str, mask_path: str | None = None) -> TFactorEvidence:
        """Full pipeline: load CT -> (segment | load mask) -> features -> T category."""
        volume = self.preprocess(ct_path)
        if mask_path is not None:
            mask = self._load_array(mask_path)
        else:
            mask = self.segment(volume)

        radiomics = self.extract_radiomics(volume, mask)
        t_category, confidence = self.classify_t_factor(radiomics, mask)

        return TFactorEvidence(
            tumour_size_mm=radiomics.get("max_diameter_mm"),
            tumour_volume_mm3=radiomics.get("volume_mm3"),
            t_category=t_category,
            confidence=confidence,
            radiomics_features=radiomics,
            segmentation_path=str(mask_path) if mask_path else None,
            metadata={
                "spacing_zyx": list(self.spacing),
                "voxel_count": radiomics.get("voxel_count"),
            },
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def preprocess(self, ct_path: str) -> np.ndarray:
        """Load a (pre-processed) CT volume from .npy or NIfTI."""
        return self._load_array(ct_path)

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Run the locally-trained nnU-Net; return a binary tumour mask.

        `volume` must be RAW HU (nnU-Net normalises internally). Wraps it in a
        SimpleITK image using self.spacing and delegates to agents.vision.inference.
        Returns a zero mask (with a warning) if no trained model is present.
        """
        import SimpleITK as sitk
        from agents.vision.inference import run_nnunet_inference

        image = sitk.GetImageFromArray(np.asarray(volume))
        sz, sy, sx = self.spacing
        image.SetSpacing((sx, sy, sz))   # SimpleITK expects (x, y, z)
        mask, _ = run_nnunet_inference(
            image, self.model_folder, device=self.config.get("device", "cpu")
        )
        return mask

    def extract_radiomics(self, volume: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Shape features from the mask. Full PyRadiomics vector is added if the
        library is installed (lazy); otherwise the basic shape features suffice
        for T classification.

        Size is measured on the PRIMARY tumour (largest connected component) using
        the configured `diameter_method` — see __init__ for why this avoids
        over-staging. Both the axial and 3D diameters are recorded for audit.
        """
        full_b = np.asarray(mask, dtype=bool)
        primary_b, n_components, largest_frac = self._primary_component(full_b)
        voxel_vol_mm3 = float(np.prod(self.spacing))
        diam_3d = self._max_diameter_mm(primary_b)
        diam_axial = self._max_axial_diameter_mm(primary_b)
        chosen = diam_axial if self.diameter_method == "axial" else diam_3d
        feats: dict[str, float] = {
            "voxel_count": int(primary_b.sum()),
            "volume_mm3": float(primary_b.sum() * voxel_vol_mm3),
            "max_diameter_mm": chosen,
            "max_diameter_3d_mm": diam_3d,
            "max_axial_diameter_mm": diam_axial,
            "n_components": int(n_components),
            "largest_component_frac": round(float(largest_frac), 4),
        }
        mask_b = primary_b

        # Optional: full PyRadiomics shape features when available.
        try:
            from radiomics import featureextractor
            import SimpleITK as sitk

            sx, sy, sz = self.spacing[2], self.spacing[1], self.spacing[0]
            img = sitk.GetImageFromArray(np.asarray(volume, dtype="float32"))
            msk = sitk.GetImageFromArray(mask_b.astype("uint8"))
            for im in (img, msk):
                im.SetSpacing((sx, sy, sz))
            extractor = featureextractor.RadiomicsFeatureExtractor()
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName("shape")
            result = extractor.execute(img, msk)
            feats.update({
                k: float(v) for k, v in result.items()
                if k.startswith("original_") and np.isscalar(v)
            })
            # Prefer PyRadiomics' own diameter for the chosen method (axial =
            # Maximum2DDiameterSlice, 3D = Maximum3DDiameter) when available.
            if self.diameter_method == "axial" and "original_shape_Maximum2DDiameterSlice" in feats:
                feats["max_diameter_mm"] = feats["original_shape_Maximum2DDiameterSlice"]
            elif self.diameter_method != "axial" and "original_shape_Maximum3DDiameter" in feats:
                feats["max_diameter_mm"] = feats["original_shape_Maximum3DDiameter"]
        except Exception as exc:  # pyradiomics not installed / extraction failed
            logger.debug("PyRadiomics unavailable, using basic shape features (%s)", exc)

        return feats

    def classify_t_factor(
        self,
        radiomics: dict[str, float],
        mask: np.ndarray,
        invasion: dict[str, bool] | None = None,
    ) -> tuple[str, float]:
        """Map greatest tumour dimension -> IASLC T category (+ optional invasion
        upstaging). Returns (t_category, confidence)."""
        primary_b, _, _ = self._primary_component(np.asarray(mask, dtype=bool))
        diam = (radiomics or {}).get("max_diameter_mm")
        if diam is None:
            diam = self._greatest_dimension_mm(primary_b)
        voxels = int(primary_b.sum())

        if voxels < self.min_tumour_voxels or diam <= 0:
            return "TX", 0.0   # no usable tumour segmentation

        t = self._size_to_t(diam)
        confidence = self._size_confidence(diam)
        t = self._apply_invasion(t, invasion)
        return t, round(confidence, 3)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _primary_component(self, mask: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Return (largest-connected-component mask, n_components, largest_frac).

        The GTV can contain disconnected nodal/satellite deposits; clinical T is
        the primary tumour, so we keep the single largest 3D-connected component.
        Disabled via use_largest_component=False (returns the mask unchanged).
        """
        mask_b = np.asarray(mask, dtype=bool)
        total = int(mask_b.sum())
        if total == 0:
            return mask_b, 0, 0.0
        if not self.use_largest_component:
            return mask_b, 1, 1.0
        from scipy.ndimage import label
        labels, n = label(mask_b)
        if n <= 1:
            return mask_b, int(n), 1.0
        counts = np.bincount(labels.ravel())
        counts[0] = 0                      # background
        largest = int(counts.argmax())
        primary = labels == largest
        return primary, int(n), counts[largest] / total

    def _greatest_dimension_mm(self, mask: np.ndarray) -> float:
        """Greatest dimension via the configured method (axial or full 3D)."""
        if self.diameter_method == "axial":
            return self._max_axial_diameter_mm(mask)
        return self._max_diameter_mm(mask)

    def _max_diameter_mm(self, mask: np.ndarray) -> float:
        """Greatest 3D dimension (mm) — max distance between any two tumour voxels."""
        coords = np.argwhere(np.asarray(mask, dtype=bool))
        if len(coords) < 2:
            return 0.0
        phys = coords * np.asarray(self.spacing, dtype=float)
        from scipy.spatial.distance import pdist
        if len(coords) < 5:
            return float(pdist(phys).max())
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(phys)
            return float(pdist(phys[hull.vertices]).max())
        except Exception:
            # degenerate (collinear/coplanar) -> bounding-box diagonal
            extent = phys.max(axis=0) - phys.min(axis=0)
            return float(np.linalg.norm(extent))

    def _max_axial_diameter_mm(self, mask: np.ndarray) -> float:
        """Greatest IN-PLANE (axial) dimension (mm) — the clinical convention.

        For each axial slice, the longest distance between two tumour voxels
        within that slice (using the y/x spacing); the tumour's greatest axial
        dimension is the max over slices. Avoids the cranio-caudal inflation of
        the 3D diagonal.
        """
        mask_b = np.asarray(mask, dtype=bool)
        if mask_b.sum() < 2:
            return 0.0
        sy, sx = float(self.spacing[1]), float(self.spacing[2])
        from scipy.spatial.distance import pdist
        best = 0.0
        for z in np.flatnonzero(mask_b.any(axis=(1, 2))):
            pts = np.argwhere(mask_b[z])            # (y, x) within the slice
            if len(pts) < 2:
                continue
            phys = pts * np.asarray([sy, sx], dtype=float)
            if len(pts) < 5:
                d = float(pdist(phys).max())
            else:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(phys)
                    d = float(pdist(phys[hull.vertices]).max())
                except Exception:
                    extent = phys.max(axis=0) - phys.min(axis=0)
                    d = float(np.linalg.norm(extent))
            if d > best:
                best = d
        return best

    @staticmethod
    def _size_to_t(diam_mm: float) -> str:
        for upper, cat in T_SIZE_THRESHOLDS:
            if diam_mm <= upper:
                return cat
        return "T4"

    @staticmethod
    def _size_confidence(diam_mm: float) -> float:
        """Higher confidence the further the diameter sits from a T boundary."""
        margin = min(abs(diam_mm - b) for b in _T_BOUNDS)
        return 0.5 + 0.4 * min(margin / 5.0, 1.0)

    @staticmethod
    def _apply_invasion(t: str, invasion: dict[str, bool] | None) -> str:
        """Upstage T from invasion flags (optional; not inferred from mask in v1)."""
        if not invasion:
            return t

        def at_least(cur: str, floor: str) -> str:
            return floor if _T_ORDER.index(floor) > _T_ORDER.index(cur) else cur

        if any(invasion.get(k) for k in
               ("mediastinal", "great_vessel", "carina", "vertebra", "trachea", "esophagus")):
            return "T4"
        if any(invasion.get(k) for k in ("chest_wall", "pericardium", "phrenic")):
            return at_least(t, "T3")
        if any(invasion.get(k) for k in ("visceral_pleura", "main_bronchus")):
            return at_least(t, "T2a")
        return t

    def _load_array(self, path: str) -> np.ndarray:
        p = Path(path)
        if p.suffix == ".npy":
            return np.load(p)
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))

    def __repr__(self) -> str:  # noqa: D105
        return f"VisionAgent(spacing_zyx={self.spacing}, model={self.model_folder.name})"
