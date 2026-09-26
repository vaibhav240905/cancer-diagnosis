import logging
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import ISIC2018Dataset
from model_unet import DiceLoss, UNet
from utils import (
    create_roi_crop,
    ensure_dir,
    get_device,
    get_segmentation_transforms,
    load_state_dict,
    save_state_dict,
    set_seed,
    setup_logging,
    visualize_pipeline_sample,
    dice_score_from_logits,
)


LOGGER = logging.getLogger(__name__)


def combined_segmentation_loss(logits, masks, bce_loss, dice_loss):
    probs = torch.sigmoid(logits)
    bce_val = bce_loss(logits, masks)
    dice_val = dice_loss(probs, masks)
    total = bce_val + dice_val
    return total, bce_val.item(), dice_val.item()


def train_one_epoch(model, loader, optimizer, bce_loss, dice_loss, device):
    model.train()
    running_total = 0.0
    running_bce = 0.0
    running_dice = 0.0
    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, bce_val, dice_val = combined_segmentation_loss(logits, masks, bce_loss, dice_loss)
        loss.backward()
        optimizer.step()

        running_total += loss.item()
        running_bce += bce_val
        running_dice += dice_val

    n = max(len(loader), 1)
    return running_total / n, running_bce / n, running_dice / n


@torch.no_grad()
def validate_one_epoch(model, loader, bce_loss, dice_loss, device):
    model.eval()
    running_total = 0.0
    running_dice_loss = 0.0
    running_dice_score = 0.0
    for images, masks in tqdm(loader, desc="Val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        total, _, dice_val = combined_segmentation_loss(logits, masks, bce_loss, dice_loss)
        running_total += total.item()
        running_dice_loss += dice_val
        running_dice_score += dice_score_from_logits(logits, masks)
    n = max(len(loader), 1)
    return running_total / n, running_dice_loss / n, running_dice_score / n


@torch.no_grad()
def generate_roi_crops(model, device):
    ensure_dir(config.ROI_OUTPUT_DIR)
    ensure_dir(config.FIGURE_DIR)
    image_names = sorted(os.listdir(config.ISIC2020_IMG_DIR))

    manifest = []
    val_tf = get_segmentation_transforms(config.IMAGE_SIZE, train=False)

    preview_done = False
    for image_name in tqdm(image_names, desc="Generating ROI crops"):
        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_id = os.path.splitext(image_name)[0]
        img_path = os.path.join(config.ISIC2020_IMG_DIR, image_name)
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            manifest.append({"isic_id": image_id, "status": "skip_bad_image", "crop_path": ""})
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        inp = val_tf(image=image_rgb)["image"].unsqueeze(0).to(device)
        logits = model(inp)
        mask_prob_small = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask_prob = cv2.resize(mask_prob_small, (w, h), interpolation=cv2.INTER_LINEAR)

        crop_rgb, meta = create_roi_crop(
            image_rgb=image_rgb,
            mask_prob=mask_prob,
            threshold=config.MASK_THRESHOLD,
            margin_ratio=config.ROI_MARGIN_RATIO,
            output_size=config.IMAGE_SIZE,
        )
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        crop_path = os.path.join(config.ROI_OUTPUT_DIR, f"{image_id}.jpg")
        cv2.imwrite(crop_path, crop_bgr)

        if not preview_done:
            preview_path = os.path.join(config.FIGURE_DIR, "segmentation_roi_preview.png")
            visualize_pipeline_sample(
                original_rgb=image_rgb,
                mask_prob=mask_prob,
                crop_rgb=crop_rgb,
                save_path=preview_path,
            )
            preview_done = True

        record = {"isic_id": image_id, "status": meta["status"], "crop_path": crop_path}
        manifest.append(record)

    manifest_df = pd.DataFrame(manifest)
    ensure_dir(config.REPORT_DIR)
    manifest_path = os.path.join(config.REPORT_DIR, "roi_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    LOGGER.info("ROI generation done. Manifest: %s", manifest_path)


def main():
    setup_logging()
    set_seed(config.SEED)
    device = get_device(config.DEVICE)
    LOGGER.info("Using device: %s", device)

    ensure_dir(config.CHECKPOINT_DIR)
    ensure_dir(config.OUTPUT_DIR)
    ensure_dir(config.FIGURE_DIR)
    ensure_dir(config.REPORT_DIR)

    base_dataset = ISIC2018Dataset(
        image_dir=config.ISIC2018_IMG_DIR,
        mask_dir=config.ISIC2018_MASK_DIR,
        transform=None,
        limit=config.LIMIT,
        strict=False,
    )
    all_image_names = base_dataset.images
    train_ids, val_ids = train_test_split(
        all_image_names,
        test_size=config.SEG_VAL_SPLIT,
        random_state=config.SEED,
    )
    LOGGER.info("Seg split -> train: %d | val: %d", len(train_ids), len(val_ids))

    train_ds = ISIC2018Dataset(
        image_dir=config.ISIC2018_IMG_DIR,
        mask_dir=config.ISIC2018_MASK_DIR,
        transform=get_segmentation_transforms(config.IMAGE_SIZE, train=True),
        sample_ids=train_ids,
        strict=False,
    )
    val_ds = ISIC2018Dataset(
        image_dir=config.ISIC2018_IMG_DIR,
        mask_dir=config.ISIC2018_MASK_DIR,
        transform=get_segmentation_transforms(config.IMAGE_SIZE, train=False),
        sample_ids=val_ids,
        strict=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    sample_images, sample_masks = next(iter(train_loader))
    LOGGER.info(
        "Sanity shapes | images=%s masks=%s",
        tuple(sample_images.shape),
        tuple(sample_masks.shape),
    )

    model = UNet(in_channels=3, out_channels=1, init_features=config.UNET_INIT_FEATURES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        tr_total, tr_bce, tr_dice = train_one_epoch(model, train_loader, optimizer, bce_loss, dice_loss, device)
        val_total, val_dice_loss, val_dice_score = validate_one_epoch(
            model, val_loader, bce_loss, dice_loss, device
        )

        LOGGER.info(
            "Epoch %d/%d | train_total=%.4f train_bce=%.4f train_dice=%.4f | val_total=%.4f val_dice_loss=%.4f val_dice_score=%.4f",
            epoch,
            config.EPOCHS,
            tr_total,
            tr_bce,
            tr_dice,
            val_total,
            val_dice_loss,
            val_dice_score,
        )

        save_state_dict(model, config.SEG_LAST_CKPT)
        if val_total < best_val_loss:
            best_val_loss = val_total
            patience_counter = 0
            save_state_dict(model, config.SEG_BEST_CKPT)
            LOGGER.info("Best segmentation model saved: %s", config.SEG_BEST_CKPT)
        else:
            patience_counter += 1
            if patience_counter > config.PATIENCE:
                LOGGER.info("Early stopping triggered at epoch %d", epoch)
                break

    if os.path.isfile(config.SEG_BEST_CKPT):
        model = load_state_dict(model, config.SEG_BEST_CKPT, device)
        LOGGER.info("Loaded best segmentation model for ROI crop generation.")
    generate_roi_crops(model, device)


if __name__ == "__main__":
    main()
