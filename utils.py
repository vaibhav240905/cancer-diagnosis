import logging
import os
import random
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_segmentation_transforms(image_size: int, train: bool = True):
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=30, p=0.4),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def get_classification_transforms(image_size: int, train: bool = True):
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()


def compute_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion_matrix(y_true, y_prob, threshold: float, save_path: str, title: str):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path: str, title: str):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) < 2:
        plt.text(0.1, 0.5, "ROC curve unavailable:\nvalidation set has one class only.", fontsize=11)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="gray", label="Random baseline")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, save_path: str, title: str):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    plt.figure(figsize=(6, 5))
    if len(np.unique(y_true)) < 2:
        plt.text(0.1, 0.5, "PR curve unavailable:\nvalidation set has one class only.", fontsize=11)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_history(history: Dict[str, list], save_prefix: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation Accuracy")
    plt.plot(epochs, history["train_f1"], marker="o", label="Train F1")
    plt.plot(epochs, history["val_f1"], marker="o", label="Validation F1")
    plt.title("Training vs Validation Accuracy / F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_acc_f1_curve.png", dpi=150)
    plt.close()


def center_crop_square(image_rgb: np.ndarray, output_size: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    crop = image_rgb[y1 : y1 + side, x1 : x1 + side]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop


def create_roi_crop(
    image_rgb: np.ndarray,
    mask_prob: np.ndarray,
    threshold: float,
    margin_ratio: float,
    output_size: int,
) -> Tuple[np.ndarray, Dict]:
    mask_bin = (mask_prob >= threshold).astype(np.uint8)
    coords = np.column_stack(np.where(mask_bin > 0))
    if coords.size == 0:
        return center_crop_square(image_rgb, output_size), {"status": "fallback_center_crop"}

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return center_crop_square(image_rgb, output_size), {"status": "fallback_center_crop"}

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    margin = int(max(w, h) * margin_ratio)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image_rgb.shape[1], x + w + margin)
    y2 = min(image_rgb.shape[0], y + h + margin)
    if x2 <= x1 or y2 <= y1:
        return center_crop_square(image_rgb, output_size), {"status": "fallback_center_crop"}

    crop = image_rgb[y1:y2, x1:x2]
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop, {"status": "roi_bbox", "bbox": [int(x1), int(y1), int(x2), int(y2)]}


def save_state_dict(model: torch.nn.Module, save_path: str):
    ensure_dir(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)


def load_state_dict(model: torch.nn.Module, load_path: str, device: torch.device):
    state = torch.load(load_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def visualize_pipeline_sample(
    original_rgb: np.ndarray,
    mask_prob: np.ndarray,
    crop_rgb: np.ndarray,
    pred_label: Optional[int] = None,
    pred_prob: Optional[float] = None,
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(12, 3.8))
    plt.subplot(1, 4, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(mask_prob, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(crop_rgb)
    plt.title("ROI Crop")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(crop_rgb)
    if pred_label is None:
        title = "Prediction"
    else:
        title = f"Pred: {pred_label} ({pred_prob:.3f})"
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, 0].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_cam_on_image(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay
