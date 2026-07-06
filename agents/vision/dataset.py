# Dataset class for loading preprocessed CT volumes and GTV masks

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RadiomicsPreprocessedDataset(Dataset):
    """Loads preprocessed CT volumes and GTV masks from .npy files.

    Expects {patient_id}_ct.npy, {patient_id}_mask.npy and
    {patient_id}_meta.json under processed_dir (written by preprocess.py).
    Returns full volumes since we do whole-volume inference.
    """

    def __init__(self, patient_ids, processed_dir):
        self.patient_ids = list(patient_ids)
        self.processed_dir = Path(processed_dir)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]

        ct = np.load(self.processed_dir / f"{patient_id}_ct.npy")
        mask = np.load(self.processed_dir / f"{patient_id}_mask.npy")

        return {
            "ct": torch.from_numpy(ct).float(),
            "mask": torch.from_numpy(mask).long(),
            "patient_id": patient_id,
            "shape": tuple(ct.shape),
        }

    def get_metadata(self, patient_id):
        """Load the preprocessing metadata JSON for a patient."""
        with open(self.processed_dir / f"{patient_id}_meta.json") as f:
            return json.load(f)
