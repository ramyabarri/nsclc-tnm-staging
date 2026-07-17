"""Vision agent.

This agent handles the imaging side and produces the T factor. It takes a CT
volume and a tumour mask (either one I pass in, or one the locally-trained nnU-Net
produces), measures the tumour, and turns its greatest dimension into an IASLC T
category.

One limitation worth stating: T can also be raised by invasion into nearby
structures (pleura, chest wall, mediastinum, main bronchus), and you cannot tell
that from a plain tumour mask. I allow those invasion flags as optional inputs but
do not try to infer them from the mask; picking them up automatically (for example
with TotalSegmentator for anatomical context) is left for future work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# (z, y, x) voxel spacing in mm, matching the preprocessing pipeline.
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
    """Estimates the T factor from a CT scan and its tumour mask.

    Example:
        agent = VisionAgent(config)
        evidence = agent.run(ct_path, mask_path=gtv_mask_path)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.spacing = tuple(self.config.get("spacing_zyx", DEFAULT_SPACING_ZYX))
        self.model_folder = Path(self.config.get("nnunet_model", DEFAULT_NNUNET_MODEL_FOLDER))
        self.min_tumour_voxels = int(self.config.get("min_tumour_voxels", 10))
        # Clinically, T is the greatest dimension of the primary tumour measured on
        # the axial (in-plane) view. If I instead take the max 3D diameter over the
        # whole GTV the number comes out too big for two reasons: the top-to-bottom
        # diagonal stretches it, and any separate nodal or satellite blobs in the
        # GTV get counted too. The defaults below avoid both. Setting
        # diameter_method="diameter3d" and use_largest_component=False gives back
        # the old whole-GTV behaviour if I want to compare.
        self.diameter_method = self.config.get("diameter_method", "axial")  # "axial" | "diameter3d"
        self.use_largest_component = bool(self.config.get("use_largest_component", True))

    def load_models(self) -> None:
        """Nothing to do here; nnU-Net is only loaded when segment() needs it."""
        return None

    def run(self, ct_path: str, mask_path: str | None = None) -> TFactorEvidence:
        """Whole pipeline: load the CT, get a mask, measure it, decide T."""
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

    def preprocess(self, ct_path: str) -> np.ndarray:
        """Load a CT volume from a .npy or NIfTI file."""
        return self._load_array(ct_path)

    def segment(self, volume: np.ndarray) -> np.ndarray:
        """Run the trained nnU-Net on the volume and return a binary tumour mask.

        The volume must be raw HU values (nnU-Net does its own normalisation). I
        wrap it in a SimpleITK image with the right spacing and hand it to
        agents.vision.inference. If no trained model is found it returns an empty
        mask and logs a warning.
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
        """Measure the tumour: volume, and both axial and 3D diameters.

        These basic shape numbers are all the T rule needs. If PyRadiomics happens
        to be installed I add its full shape feature set on top. Everything is
        measured on the primary tumour (the largest connected component) for the
        reason explained in __init__, and I keep both diameters so the choice can
        be checked later.
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
        """Turn the greatest tumour dimension into a T category.

        Optionally raises T if invasion flags are supplied. Returns the category
        and a confidence, or ('TX', 0.0) if there is no usable tumour to measure.
        """
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

    def _primary_component(self, mask: np.ndarray) -> tuple[np.ndarray, int, float]:
        """Keep only the largest connected blob in the mask (the primary tumour).

        Returns that blob, how many blobs there were, and what fraction of the
        voxels the largest one holds. The GTV can include separate nodal or
        satellite bits, but clinical T is about the primary, so I drop the rest.
        Turned off by use_largest_component=False, which returns the mask as-is.
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
        """Greatest 3D dimension in mm: the max distance between any two tumour voxels."""
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
        """Greatest in-plane (axial) dimension in mm, the clinical convention.

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
