"""CLI for batch CT preprocessing of NSCLC-Radiomics patients.

Usage
-----
    python -m scripts.preprocessing.batch_preprocess --patients LUNG1-001 LUNG1-013
    python -m scripts.preprocessing.batch_preprocess --all
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm.auto import tqdm

from scripts.preprocessing.preprocess import preprocess_patient

RAW_DIR = Path("data/raw/NSCLC-Radiomics")
OUTPUT_DIR = Path("data/processed/radiomics")
TARGET_SPACING = (1.0, 1.0, 3.0)


def discover_patients(raw_dir: Path) -> list[str]:
    """List all patient IDs (subdirectory names) under ``raw_dir``."""
    return sorted(p.name for p in raw_dir.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-preprocess NSCLC-Radiomics CT scans.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--patients", nargs="+", help="Patient IDs to preprocess, e.g. LUNG1-001 LUNG1-013")
    group.add_argument("--all", action="store_true", help="Preprocess every patient under --raw-dir")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Root directory of NSCLC-Radiomics")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to write processed outputs")
    args = parser.parse_args()

    patients = args.patients if args.patients else discover_patients(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[dict[str, str]] = []
    for patient_id in tqdm(patients, desc="Preprocessing patients"):
        try:
            preprocess_patient(
                patient_id=patient_id,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                target_spacing=TARGET_SPACING,
            )
        except Exception as exc:
            failed.append({"patient_id": patient_id, "error": str(exc)})

    if failed:
        failed_path = args.output_dir / "_failed.csv"
        with open(failed_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "error"])
            writer.writeheader()
            writer.writerows(failed)
        print(f"{len(failed)} patient(s) failed; see {failed_path}")

    succeeded = len(patients) - len(failed)
    print(f"Done: {succeeded}/{len(patients)} patients processed successfully.")


if __name__ == "__main__":
    main()
