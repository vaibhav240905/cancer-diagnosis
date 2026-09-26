import os
import torch

# -------------------------
# Required dataset paths
# -------------------------
ISIC2018_IMG_DIR = r"F:\SkinCancerProject\data\isic2018\images"
ISIC2018_MASK_DIR = r"F:\SkinCancerProject\data\isic2018\masks"

ISIC2020_IMG_DIR = r"F:\SkinCancerProject\data\isic2020\images"
ISIC2020_CSV = r"F:\SkinCancerProject\data\isic2020\train.csv"

PROJECT_ROOT = r"C:\Users\Saurabh\Desktop\PE2_Project"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# -------------------------
# Runtime / hardware settings
# -------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 15
PATIENCE = 3
NUM_WORKERS = 0  # Windows-friendly
SEED = 42
LIMIT = None  # Example: set to 1000 for low-RAM quick runs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Segmentation settings
# -------------------------
SEG_VAL_SPLIT = 0.2
MASK_THRESHOLD = 0.5
ROI_MARGIN_RATIO = 0.08
UNET_INIT_FEATURES = 32

# -------------------------
# Classification settings
# -------------------------
CLF_VAL_SPLIT = 0.2
CLF_THRESHOLD = 0.5
USE_POS_WEIGHT = True

# Comparison track
RUN_ROI_EXPERIMENT = True
RUN_FULL_IMAGE_BASELINE = True

# -------------------------
# Output structure
# -------------------------
ROI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "roi_crops")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# -------------------------
# Checkpoint names (.pth)
# -------------------------
SEG_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "unet_best.pth")
SEG_LAST_CKPT = os.path.join(CHECKPOINT_DIR, "unet_last.pth")

CLF_ROI_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "efficientnet_b0_roi_best.pth")
CLF_FULL_BEST_CKPT = os.path.join(CHECKPOINT_DIR, "efficientnet_b0_full_best.pth")
