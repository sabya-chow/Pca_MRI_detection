"""Data pipeline: DICOM loading, normalisation, PyTorch Datasets."""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

import pydicom
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# ── constants ─────────────────────────────────────────────────────────────────
SEQUENCES   = ["AX_T2", "AX_DIFFUSION_ADC", "AX_DIFFUSION_CALC_BVAL"]
TARGET_H    = 128
TARGET_W    = 128
TARGET_D    = 30   # slices per volume (pad/crop to this)


# ── low-level DICOM utils ─────────────────────────────────────────────────────

def _sort_key(path: Path) -> int:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return int(getattr(ds, "InstanceNumber", 0))
    except Exception:
        return 0


def load_volume(patient_id: str, sequence: str, dicom_root) -> Optional[np.ndarray]:
    """Load one sequence for one patient.

    Returns float32 ndarray of shape (D, H, W), or None if missing.
    For TRACEW (60 files = 2 b-values × 30 slices) takes every other file.
    """
    seq_path = Path(dicom_root) / patient_id / sequence
    if not seq_path.exists():
        return None
    files = sorted(seq_path.glob("*.dcm"), key=_sort_key)
    if not files:
        return None

    # TRACEW has 60 files (b=0 and b=1000 interleaved); keep one set
    if len(files) > 35:
        files = files[::2]     # every other → 30 slices

    slices = []
    for f in files:
        try:
            arr = pydicom.dcmread(str(f)).pixel_array.astype(np.float32)
            slices.append(arr)
        except Exception:
            continue
    if not slices:
        return None
    return np.stack(slices, axis=0)   # (D, H, W)


def preprocess_volume(vol: np.ndarray,
                      target_d: int = TARGET_D,
                      target_h: int = TARGET_H,
                      target_w: int = TARGET_W) -> np.ndarray:
    """Resize spatially, pad/crop depth, z-score normalise.

    Returns float32 (target_d, target_h, target_w).
    """
    D, H, W = vol.shape

    # Resize H × W for each slice
    resized = np.stack(
        [cv2.resize(sl, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
         for sl in vol],
        axis=0
    )  # (D, target_h, target_w)

    # Depth: crop or pad to target_d
    cur_d = resized.shape[0]
    if cur_d >= target_d:
        start = (cur_d - target_d) // 2
        resized = resized[start: start + target_d]
    else:
        pad = target_d - cur_d
        pad_before = pad // 2
        pad_after  = pad - pad_before
        resized = np.pad(resized, ((pad_before, pad_after), (0, 0), (0, 0)))

    # Z-score normalise per volume
    mu, sigma = resized.mean(), resized.std() + 1e-6
    resized = (resized - mu) / sigma
    return resized.astype(np.float32)


# ── Volume-level dataset (for 3-D classifier) ─────────────────────────────────

class ProstateMRIDataset(Dataset):
    """One item = (C × D × H × W tensor, ordinal_label [0-4]).

    channels: list of sequences to stack. Single sequence → C=1.
    label: exam_level - 1  (0-indexed so PIRADS-1 → 0, PIRADS-5 → 4).
    """

    def __init__(
        self,
        patient_ids: List[int],
        labels_df: pd.DataFrame,
        dicom_root,
        sequences: List[str] = ("AX_T2",),
        target_d: int  = TARGET_D,
        target_h: int  = TARGET_H,
        target_w: int  = TARGET_W,
        augment: bool  = False,
    ):
        self.patient_ids = patient_ids
        self.labels      = labels_df.set_index("fastmri_pt_id")["exam_level"].to_dict()
        self.dicom_root  = Path(dicom_root)
        self.sequences   = list(sequences)
        self.td, self.th, self.tw = target_d, target_h, target_w
        self.augment     = augment

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        pid = self.patient_ids[idx]
        pt_str = f"{pid:03d}"
        label  = int(self.labels[pid]) - 1   # 0-indexed

        channels = []
        for seq in self.sequences:
            vol = load_volume(pt_str, seq, self.dicom_root)
            if vol is None:
                vol = np.zeros((self.td, self.th, self.tw), dtype=np.float32)
            else:
                vol = preprocess_volume(vol, self.td, self.th, self.tw)
            channels.append(vol)

        tensor = torch.tensor(np.stack(channels, axis=0))  # (C, D, H, W)

        if self.augment:
            tensor = self._augment(tensor)

        return tensor, label

    @staticmethod
    def _augment(x: torch.Tensor) -> torch.Tensor:
        C, D, H, W = x.shape
        # Random horizontal flip along width
        if torch.rand(1) > 0.5:
            x = x.flip(-1)
        # Slice-wise random brightness shift
        shift = torch.empty(C, D, 1, 1).uniform_(-0.1, 0.1)
        x = x + shift
        return x


# ── Slice-level dataset (for SimCLR pre-training) ────────────────────────────

class SimCLRSliceDataset(Dataset):
    """One item = (augmented_view_1, augmented_view_2) of a single T2 slice.

    Returns tensors of shape (1, target_h, target_w).
    """

    def __init__(
        self,
        slice_records: List[Tuple[str, int]],   # [(patient_str, slice_idx), ...]
        dicom_root,
        target_h: int = TARGET_H,
        target_w: int = TARGET_W,
    ):
        self.records    = slice_records
        self.dicom_root = Path(dicom_root)
        self.th = target_h
        self.tw = target_w
        self.aug = T.Compose([
            T.RandomResizedCrop((target_h, target_w), scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        ])
        self._cache: dict = {}

    def _load_slice(self, pt_str: str, slice_idx: int) -> torch.Tensor:
        key = pt_str
        if key not in self._cache:
            vol = load_volume(pt_str, "AX_T2", self.dicom_root)
            if vol is None:
                self._cache[key] = None
            else:
                vols = []
                for sl in vol:
                    sl_r = cv2.resize(sl, (self.tw, self.th), interpolation=cv2.INTER_LINEAR)
                    vols.append(sl_r)
                self._cache[key] = np.stack(vols, axis=0)  # (D, H, W)
        arr = self._cache[key]
        if arr is None:
            return torch.zeros(1, self.th, self.tw)
        sl = arr[min(slice_idx, len(arr)-1)]
        # z-score
        mu, sigma = sl.mean(), sl.std() + 1e-6
        sl = (sl - mu) / sigma
        return torch.tensor(sl, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pt_str, sl_idx = self.records[idx]
        img = self._load_slice(pt_str, sl_idx)
        # Repeat 1-channel to 3-channel for RandomResizedCrop compatibility
        img3 = img.expand(3, -1, -1)
        v1 = self.aug(img3)[[0]].contiguous()  # back to 1-channel
        v2 = self.aug(img3)[[0]].contiguous()
        return v1, v2


# ── Helper: build train/val/test patient splits ───────────────────────────────

def get_splits(vol_df: pd.DataFrame, t2_df: pd.DataFrame):
    """Return (train_ids, val_ids, test_ids) as lists of int patient IDs."""
    split_map = t2_df.groupby("fastmri_pt_id")["data_split"].first().to_dict()
    vol_clean = vol_df.dropna(subset=["exam_level"])
    train_ids = [int(r.fastmri_pt_id) for r in vol_clean.itertuples()
                 if split_map.get(r.fastmri_pt_id, "training") == "training"]
    val_ids   = [int(r.fastmri_pt_id) for r in vol_clean.itertuples()
                 if split_map.get(r.fastmri_pt_id, "training") == "validation"]
    test_ids  = [int(r.fastmri_pt_id) for r in vol_clean.itertuples()
                 if split_map.get(r.fastmri_pt_id, "training") == "test"]
    return train_ids, val_ids, test_ids


def build_simclr_records(train_ids: List[int]) -> List[Tuple[str, int]]:
    """All (patient_str, slice_idx) pairs for SimCLR training."""
    records = []
    for pid in train_ids:
        for sl in range(TARGET_D):
            records.append((f"{pid:03d}", sl))
    return records
