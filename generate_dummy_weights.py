import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import config
from model_unet import UNet

def build_classifier():
    model = efficientnet_b0()
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model

def build_unet():
    model = UNet(in_channels=3, out_channels=1, init_features=config.UNET_INIT_FEATURES)
    return model

def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    print("Generating dummy UNet weights...")
    unet = build_unet()
    torch.save(unet.state_dict(), config.SEG_BEST_CKPT)
    
    print("Generating dummy ROI classifier weights...")
    roi_clf = build_classifier()
    torch.save(roi_clf.state_dict(), config.CLF_ROI_BEST_CKPT)
    
    print("Generating dummy Full classifier weights...")
    full_clf = build_classifier()
    torch.save(full_clf.state_dict(), config.CLF_FULL_BEST_CKPT)
    
    print("Done!")

if __name__ == "__main__":
    main()
