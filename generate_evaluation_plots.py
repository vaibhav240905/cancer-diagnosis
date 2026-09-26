import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

import config
from dataset import ISIC2020Dataset, build_isic2020_records
from utils import (
    GradCAM,
    ensure_dir,
    get_classification_transforms,
    get_device,
    load_state_dict,
    overlay_cam_on_image,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    set_seed,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def build_classifier(device: torch.device):
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model.to(device)


def build_validation_records(use_crops: bool) -> List[Dict]:
    records = build_isic2020_records(
        csv_path=config.ISIC2020_CSV,
        image_root=config.ISIC2020_IMG_DIR,
        use_crops=use_crops,
        crop_root=config.ROI_OUTPUT_DIR if use_crops else None,
    )
    if config.LIMIT is not None:
        records = records[: config.LIMIT]
    if len(records) < 2:
        raise ValueError(f"Not enough samples for evaluation: use_crops={use_crops}")

    labels = [r["label"] for r in records]
    _, val_records = train_test_split(
        records,
        test_size=config.CLF_VAL_SPLIT,
        random_state=config.SEED,
        stratify=labels,
    )
    return val_records


@torch.no_grad()
def collect_validation_predictions(model, loader, device):
    model.eval()
    y_true_all = []
    y_prob_all = []
    for images, labels, _ in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

        y_true_all.extend(labels.cpu().numpy().reshape(-1).astype(int).tolist())
        y_prob_all.extend(probs.tolist())
    return y_true_all, y_prob_all


def save_gradcam_grid(model, dataset, device, save_path: str, num_samples: int = 4):
    if len(dataset) == 0:
        LOGGER.warning("Dataset is empty, skipping Grad-CAM: %s", save_path)
        return

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    n = min(num_samples, len(dataset))

    plt.figure(figsize=(10, 4 * n))
    for i in range(n):
        image_t, _, image_id = dataset[i]
        input_tensor = image_t.unsqueeze(0).to(device)
        cam = gradcam(input_tensor)

        image_np = image_t.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        image_rgb = (image_np * 255).astype(np.uint8)
        overlay = overlay_cam_on_image(image_rgb, cam)

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(image_rgb)
        plt.title(f"Input: {image_id}")
        plt.axis("off")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_for_experiment(suffix: str, ckpt_path: str, use_crops: bool, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if use_crops and not os.path.isdir(config.ROI_OUTPUT_DIR):
        raise FileNotFoundError(f"ROI directory not found: {config.ROI_OUTPUT_DIR}")

    val_records = build_validation_records(use_crops=use_crops)
    val_ds = ISIC2020Dataset(
        csv_path=config.ISIC2020_CSV,
        image_root=config.ISIC2020_IMG_DIR,
        records=val_records,
        transform=get_classification_transforms(config.IMAGE_SIZE, train=False),
        strict=False,
        return_id=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    model = build_classifier(device)
    model = load_state_dict(model, ckpt_path, device)

    y_true, y_prob = collect_validation_predictions(model, val_loader, device)

    roc_path = os.path.join(config.FIGURE_DIR, f"roc_curve_{suffix}.png")
    pr_path = os.path.join(config.FIGURE_DIR, f"pr_curve_{suffix}.png")
    cm_path = os.path.join(config.FIGURE_DIR, f"confusion_matrix_{suffix}.png")
    gc_path = os.path.join(config.FIGURE_DIR, f"gradcam_{suffix}.png")

    plot_roc_curve(y_true, y_prob, roc_path, f"ROC Curve ({suffix})")
    plot_precision_recall_curve(y_true, y_prob, pr_path, f"Precision-Recall Curve ({suffix})")
    plot_confusion_matrix(
        y_true=y_true,
        y_prob=y_prob,
        threshold=config.CLF_THRESHOLD,
        save_path=cm_path,
        title=f"Confusion Matrix ({suffix})",
    )
    save_gradcam_grid(model, val_ds, device, gc_path, num_samples=4)

    LOGGER.info("Saved plots for %s -> %s, %s, %s, %s", suffix, roc_path, pr_path, cm_path, gc_path)


def main():
    setup_logging()
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    ensure_dir(config.FIGURE_DIR)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Generating evaluation plots from saved checkpoints only (no retraining).")

    generate_for_experiment(
        suffix="roi",
        ckpt_path=config.CLF_ROI_BEST_CKPT,
        use_crops=True,
        device=device,
    )
    generate_for_experiment(
        suffix="full",
        ckpt_path=config.CLF_FULL_BEST_CKPT,
        use_crops=False,
        device=device,
    )


if __name__ == "__main__":
    main()
