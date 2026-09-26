import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
from tqdm import tqdm

import config
from dataset import ISIC2020Dataset, build_isic2020_records
from utils import (
    GradCAM,
    compute_classification_metrics,
    ensure_dir,
    get_classification_transforms,
    get_device,
    load_state_dict,
    overlay_cam_on_image,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_history,
    save_state_dict,
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


def get_train_sampler(records: List[Dict]):
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    sample_weights = np.array([1.0 / class_counts[label] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_pos_weight(records: List[Dict], device: torch.device):
    labels = np.array([r["label"] for r in records], dtype=np.int64)
    positives = max(labels.sum(), 1)
    negatives = max(len(labels) - positives, 1)
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    y_true_all = []
    y_prob_all = []

    for batch in tqdm(loader, desc="Train" if train else "Val", leave=False):
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        y_true_all.extend(labels.detach().cpu().numpy().reshape(-1).astype(int).tolist())
        y_prob_all.extend(probs.tolist())

    metrics = compute_classification_metrics(y_true_all, y_prob_all, threshold=config.CLF_THRESHOLD)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics, y_true_all, y_prob_all


def save_gradcam_examples(model, dataset, device, save_path, num_samples: int = 4):
    if len(dataset) == 0 or num_samples <= 0:
        return
    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)
    n = min(num_samples, len(dataset))
    ncols = 2
    nrows = n

    plt.figure(figsize=(10, 4 * nrows))
    for i in range(n):
        image_t, _, image_id = dataset[i]
        input_tensor = image_t.unsqueeze(0).to(device)
        cam = gradcam(input_tensor)

        image_np = image_t.permute(1, 2, 0).cpu().numpy()
        # Undo ImageNet normalization for visualization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_np = (image_np * std + mean)
        image_np = np.clip(image_np, 0, 1)
        image_rgb = (image_np * 255).astype(np.uint8)
        overlay = overlay_cam_on_image(image_rgb, cam)

        plt.subplot(nrows, ncols, 2 * i + 1)
        plt.imshow(image_rgb)
        plt.title(f"Input: {image_id}")
        plt.axis("off")

        plt.subplot(nrows, ncols, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("Grad-CAM Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_experiment(use_crops: bool, ckpt_path: str, suffix: str, device: torch.device):
    records = build_isic2020_records(
        csv_path=config.ISIC2020_CSV,
        image_root=config.ISIC2020_IMG_DIR,
        use_crops=use_crops,
        crop_root=config.ROI_OUTPUT_DIR if use_crops else None,
    )
    if config.LIMIT is not None:
        records = records[: config.LIMIT]
    if len(records) < 2:
        raise ValueError(f"Not enough records for experiment: {suffix}")

    labels = [r["label"] for r in records]
    train_records, val_records = train_test_split(
        records,
        test_size=config.CLF_VAL_SPLIT,
        random_state=config.SEED,
        stratify=labels,
    )
    LOGGER.info("%s split -> train: %d | val: %d", suffix, len(train_records), len(val_records))

    train_ds = ISIC2020Dataset(
        csv_path=config.ISIC2020_CSV,
        image_root=config.ISIC2020_IMG_DIR,
        records=train_records,
        transform=get_classification_transforms(config.IMAGE_SIZE, train=True),
        strict=False,
        return_id=True,
    )
    val_ds = ISIC2020Dataset(
        csv_path=config.ISIC2020_CSV,
        image_root=config.ISIC2020_IMG_DIR,
        records=val_records,
        transform=get_classification_transforms(config.IMAGE_SIZE, train=False),
        strict=False,
        return_id=True,
    )

    train_sampler = get_train_sampler(train_records)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    x0, y0, _ = next(iter(train_loader))
    LOGGER.info("%s sanity shapes | images=%s labels=%s", suffix, tuple(x0.shape), tuple(y0.shape))

    model = build_classifier(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    if config.USE_POS_WEIGHT:
        pos_weight = get_pos_weight(train_records, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_metrics = None
    best_y_true = None
    best_y_prob = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "train_f1": [],
        "val_f1": [],
    }

    for epoch in range(1, config.EPOCHS + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics, y_true, y_prob = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        LOGGER.info(
            "[%s] Epoch %d/%d | train_loss=%.4f val_loss=%.4f val_acc=%.4f val_prec=%.4f val_rec=%.4f val_f1=%.4f val_auc=%.4f",
            suffix,
            epoch,
            config.EPOCHS,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["roc_auc"],
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            save_state_dict(model, ckpt_path)
            best_metrics = val_metrics
            best_y_true = y_true
            best_y_prob = y_prob
            LOGGER.info("[%s] Best model saved: %s", suffix, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter > config.PATIENCE:
                LOGGER.info("[%s] Early stopping at epoch %d", suffix, epoch)
                break

    model = load_state_dict(model, ckpt_path, device)

    cm_path = os.path.join(config.FIGURE_DIR, f"confusion_matrix_{suffix}.png")
    plot_confusion_matrix(
        y_true=best_y_true,
        y_prob=best_y_prob,
        threshold=config.CLF_THRESHOLD,
        save_path=cm_path,
        title=f"Confusion Matrix ({suffix})",
    )

    roc_path = os.path.join(config.FIGURE_DIR, f"roc_curve_{suffix}.png")
    pr_path = os.path.join(config.FIGURE_DIR, f"precision_recall_curve_{suffix}.png")
    plot_roc_curve(best_y_true, best_y_prob, roc_path, f"ROC Curve ({suffix})")
    plot_precision_recall_curve(best_y_true, best_y_prob, pr_path, f"Precision-Recall Curve ({suffix})")

    history_prefix = os.path.join(config.FIGURE_DIR, f"training_{suffix}")
    plot_training_history(history, history_prefix)

    gradcam_path = os.path.join(config.FIGURE_DIR, f"gradcam_{suffix}.png")
    save_gradcam_examples(model, val_ds, device, gradcam_path, num_samples=4)
    return best_metrics


def main():
    setup_logging()
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    LOGGER.info("Using device: %s", device)

    ensure_dir(config.CHECKPOINT_DIR)
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.FIGURE_DIR)
    ensure_dir(config.REPORT_DIR)

    results = {}
    if config.RUN_ROI_EXPERIMENT:
        if not os.path.isdir(config.ROI_OUTPUT_DIR):
            raise FileNotFoundError(
                f"ROI directory not found: {config.ROI_OUTPUT_DIR}. Run train_segmentation.py first."
            )
        results["roi"] = run_experiment(
            use_crops=True,
            ckpt_path=config.CLF_ROI_BEST_CKPT,
            suffix="roi",
            device=device,
        )

    if config.RUN_FULL_IMAGE_BASELINE:
        results["full"] = run_experiment(
            use_crops=False,
            ckpt_path=config.CLF_FULL_BEST_CKPT,
            suffix="full",
            device=device,
        )

    if results:
        report_path = os.path.join(config.REPORT_DIR, "comparison_metrics.csv")
        rows = []
        for name, metrics in results.items():
            row = {"experiment": name}
            row.update(metrics)
            rows.append(row)
        import pandas as pd

        pd.DataFrame(rows).to_csv(report_path, index=False)
        LOGGER.info("Comparison metrics saved: %s", report_path)


if __name__ == "__main__":
    main()
