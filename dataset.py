import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SegmentationSample:
    image_id: str
    image_name: str
    image_path: str
    mask_path: str


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMG_EXTENSIONS


def build_isic2018_pairs(image_dir: str, mask_dir: str) -> List[SegmentationSample]:
    samples: List[SegmentationSample] = []
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    for image_name in sorted(os.listdir(image_dir)):
        if not is_image_file(image_name):
            continue
        image_path = os.path.join(image_dir, image_name)
        image_id = os.path.splitext(image_name)[0]
        mask_name = f"{image_id}_segmentation.png"
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.isfile(mask_path):
            LOGGER.warning("Missing mask for %s -> %s", image_name, mask_path)
            continue
        samples.append(
            SegmentationSample(
                image_id=image_id,
                image_name=image_name,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return samples


def _resolve_image_path(root_dir: str, image_id: str) -> Optional[str]:
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(root_dir, f"{image_id}{ext}")
        if os.path.isfile(path):
            return path
    return None


def build_isic2020_records(
    csv_path: str,
    image_root: str,
    use_crops: bool = False,
    crop_root: Optional[str] = None,
) -> List[Dict]:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(image_root):
        raise FileNotFoundError(f"Image root not found: {image_root}")

    df = pd.read_csv(csv_path)
    if "isic_id" not in df.columns:
        raise ValueError("Expected 'isic_id' column in classification CSV")
    if "target" not in df.columns:
        raise ValueError("Expected 'target' column in classification CSV")

    records: List[Dict] = []
    missing_count = 0
    for _, row in df.iterrows():
        image_id = str(row["isic_id"])
        label = int(row["target"])
        if use_crops:
            if crop_root is None:
                raise ValueError("crop_root is required when use_crops=True")
            image_path = _resolve_image_path(crop_root, image_id)
        else:
            image_path = _resolve_image_path(image_root, image_id)
        if image_path is None:
            missing_count += 1
            continue
        records.append({"isic_id": image_id, "image_path": image_path, "label": label})

    LOGGER.info(
        "ISIC2020 records loaded: %d | missing images skipped: %d | use_crops=%s",
        len(records),
        missing_count,
        use_crops,
    )
    return records


class ISIC2018Dataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        sample_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        strict: bool = False,
    ):
        self.transform = transform
        self.strict = strict

        samples = build_isic2018_pairs(image_dir, mask_dir)
        if sample_ids is not None:
            keep = set(sample_ids)
            samples = [s for s in samples if s.image_name in keep]
        if limit is not None:
            samples = samples[:limit]
        if len(samples) == 0:
            raise ValueError("ISIC2018Dataset has no valid samples after filtering")

        self.samples = samples
        self.images = [s.image_name for s in samples]
        LOGGER.info("ISIC2018Dataset ready with %d samples", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_bgr = cv2.imread(sample.image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or mask is None:
            message = f"Failed to read sample: image={sample.image_path} mask={sample.mask_path}"
            if self.strict:
                raise ValueError(message)
            LOGGER.warning(message)
            # Fallback to nearest safe sample
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


class ISIC2020Dataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        transform=None,
        records: Optional[List[Dict]] = None,
        use_crops: bool = False,
        crop_root: Optional[str] = None,
        limit: Optional[int] = None,
        strict: bool = False,
        return_id: bool = True,
    ):
        self.transform = transform
        self.strict = strict
        self.return_id = return_id

        if records is None:
            records = build_isic2020_records(
                csv_path=csv_path,
                image_root=image_root,
                use_crops=use_crops,
                crop_root=crop_root,
            )
        if limit is not None:
            records = records[:limit]
        if len(records) == 0:
            raise ValueError("ISIC2020Dataset has no valid records")

        self.records = records
        LOGGER.info("ISIC2020Dataset ready with %d samples", len(self.records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image_bgr = cv2.imread(rec["image_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            message = f"Failed to read image: {rec['image_path']}"
            if self.strict:
                raise ValueError(message)
            LOGGER.warning(message)
            new_idx = (idx + 1) % len(self.records)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        label = np.float32(rec["label"])

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_t = torch.tensor([label], dtype=torch.float32)
        if self.return_id:
            return image, label_t, rec["isic_id"]
        return image, label_t
