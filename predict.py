import argparse
import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

import config
from model_unet import UNet
from utils import (
    GradCAM,
    create_roi_crop,
    ensure_dir,
    get_classification_transforms,
    get_device,
    get_segmentation_transforms,
    load_state_dict,
    overlay_cam_on_image,
    set_seed,
    setup_logging,
)


LOGGER = logging.getLogger(__name__)


def build_classifier(device: torch.device):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model.to(device)


def build_unet(device: torch.device):
    model = UNet(in_channels=3, out_channels=1, init_features=config.UNET_INIT_FEATURES)
    return model.to(device)


def denormalize_image(image_t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = image_t.permute(1, 2, 0).cpu().numpy()
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    return (image_np * 255).astype(np.uint8)


@torch.no_grad()
def predict_with_classifier(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    cls_transform,
    device: torch.device,
):
    x = cls_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits).item()
    pred = int(prob >= config.CLF_THRESHOLD)
    return pred, prob, x.squeeze(0).cpu()


def gradcam_overlay(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    image_rgb_for_overlay: np.ndarray,
    device: torch.device,
):
    x = image_tensor.unsqueeze(0).to(device)
    cam_generator = GradCAM(model, model.features[-1])
    cam = cam_generator(x)
    return overlay_cam_on_image(image_rgb_for_overlay, cam)


@torch.no_grad()
def predict_mask(unet: torch.nn.Module, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    seg_tf = get_segmentation_transforms(config.IMAGE_SIZE, train=False)
    x = seg_tf(image=image_rgb)["image"].unsqueeze(0).to(device)
    logits = unet(x)
    mask_small = torch.sigmoid(logits).squeeze().cpu().numpy()
    h, w = image_rgb.shape[:2]
    return cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)


def save_roi_panel(
    save_path: str,
    original_rgb: np.ndarray,
    mask_prob: np.ndarray,
    crop_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    pred_label: int,
    pred_prob: float,
):
    plt.figure(figsize=(14, 4))
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
    plt.imshow(overlay_rgb)
    label_name = "Malignant" if pred_label == 1 else "Benign"
    plt.title(f"Grad-CAM\n{label_name}: {pred_prob:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_full_panel(
    save_path: str,
    original_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    pred_label: int,
    pred_prob: float,
):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_rgb)
    label_name = "Malignant" if pred_label == 1 else "Benign"
    plt.title(f"Grad-CAM\n{label_name}: {pred_prob:.4f}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-image prediction for skin cancer pipeline."
    )
    parser.add_argument("--image_path", type=str, required=True, help="Path to input dermoscopic image.")
    parser.add_argument(
        "--mode",
        type=str,
        default="roi",
        choices=["roi", "full", "both"],
        help="Prediction mode: roi (segmentation+crop), full (full-image classifier), or both.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=config.FIGURE_DIR,
        help="Directory where prediction figures are saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional output filename prefix. Default uses input image name.",
    )
    return parser.parse_args()


def main():
    setup_logging()
    set_seed(config.SEED)
    args = parse_args()
    device = get_device(config.DEVICE)
    ensure_dir(args.save_dir)

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Input image not found: {args.image_path}")

    image_bgr = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {args.image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    prefix = args.prefix if args.prefix else base_name

    cls_tf = get_classification_transforms(config.IMAGE_SIZE, train=False)

    if args.mode in ["roi", "both"]:
        if not os.path.isfile(config.SEG_BEST_CKPT):
            raise FileNotFoundError(f"Segmentation checkpoint not found: {config.SEG_BEST_CKPT}")
        if not os.path.isfile(config.CLF_ROI_BEST_CKPT):
            raise FileNotFoundError(f"ROI classifier checkpoint not found: {config.CLF_ROI_BEST_CKPT}")

        unet = build_unet(device)
        unet = load_state_dict(unet, config.SEG_BEST_CKPT, device)

        roi_clf = build_classifier(device)
        roi_clf = load_state_dict(roi_clf, config.CLF_ROI_BEST_CKPT, device)

        mask_prob = predict_mask(unet, image_rgb, device)
        crop_rgb, crop_meta = create_roi_crop(
            image_rgb=image_rgb,
            mask_prob=mask_prob,
            threshold=config.MASK_THRESHOLD,
            margin_ratio=config.ROI_MARGIN_RATIO,
            output_size=config.IMAGE_SIZE,
        )
        pred, prob, crop_tensor = predict_with_classifier(roi_clf, crop_rgb, cls_tf, device)
        crop_rgb_norm = denormalize_image(crop_tensor)
        overlay = gradcam_overlay(roi_clf, crop_tensor, crop_rgb_norm, device)

        save_path = os.path.join(args.save_dir, f"{prefix}_prediction_roi.png")
        save_roi_panel(save_path, image_rgb, mask_prob, crop_rgb, overlay, pred, prob)
        LOGGER.info(
            "ROI prediction | label=%s prob=%.4f | crop_status=%s | saved=%s",
            "malignant" if pred == 1 else "benign",
            prob,
            crop_meta.get("status", "unknown"),
            save_path,
        )

    if args.mode in ["full", "both"]:
        if not os.path.isfile(config.CLF_FULL_BEST_CKPT):
            raise FileNotFoundError(f"Full-image classifier checkpoint not found: {config.CLF_FULL_BEST_CKPT}")

        full_clf = build_classifier(device)
        full_clf = load_state_dict(full_clf, config.CLF_FULL_BEST_CKPT, device)

        pred, prob, image_tensor = predict_with_classifier(full_clf, image_rgb, cls_tf, device)
        image_rgb_norm = denormalize_image(image_tensor)
        overlay = gradcam_overlay(full_clf, image_tensor, image_rgb_norm, device)

        save_path = os.path.join(args.save_dir, f"{prefix}_prediction_full.png")
        save_full_panel(save_path, image_rgb, overlay, pred, prob)
        LOGGER.info(
            "Full prediction | label=%s prob=%.4f | saved=%s",
            "malignant" if pred == 1 else "benign",
            prob,
            save_path,
        )


if __name__ == "__main__":
    main()
