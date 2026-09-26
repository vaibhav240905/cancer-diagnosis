import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        features = [init_features, init_features * 2, init_features * 4, init_features * 8]

        self.down1 = DoubleConv(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(features[2], features[3])
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3] * 2, features[3])
        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2] * 2, features[2])
        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1] * 2, features[1])
        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _align(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = nn.functional.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x):
        e1 = self.down1(x)
        e2 = self.down2(self.pool1(e1))
        e3 = self.down3(self.pool2(e2))
        e4 = self.down4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self._align(self.up4(b), e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self._align(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self._align(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self._align(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(d1)  # Return logits; apply sigmoid outside when needed.
        return logits


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor):
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()
