import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SharedUNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(3, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

    def forward(self, x: torch.Tensor) -> dict:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        bottleneck = self.bottleneck(self.pool(s4))
        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "bottleneck": bottleneck}


class SpatialTemporalAttention(nn.Module):
    """Global self-attention over both temporal feature maps.

    The token axis is H * W * 2, so each position in image A can attend to
    every encoded position in both A and B.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 32)
        self.query = nn.Conv2d(channels, reduced, kernel_size=1)
        self.key = nn.Conv2d(channels, reduced, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feature_a.shape
        x = torch.cat([feature_a, feature_b], dim=0)

        query = self.query(x).reshape(2, batch, -1, height * width).permute(1, 0, 3, 2)
        key = self.key(x).reshape(2, batch, -1, height * width).permute(1, 0, 3, 2)
        value = self.value(x).reshape(2, batch, channels, height * width).permute(1, 0, 3, 2)

        query = query.reshape(batch, 2 * height * width, -1)
        key = key.reshape(batch, 2 * height * width, -1)
        value = value.reshape(batch, 2 * height * width, channels)

        attention = torch.softmax((query @ key.transpose(1, 2)) / math.sqrt(key.shape[-1]), dim=-1)
        out = attention @ value
        out = out.reshape(batch, 2, height * width, channels).permute(1, 0, 3, 2)
        out = out.reshape(2 * batch, channels, height, width)
        out = x + self.gamma * self.proj(out)
        return out[:batch], out[batch:]


class FusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([feature_a, feature_b, torch.abs(feature_a - feature_b)], dim=1))


class SiameseTemporalAttentionUNet(nn.Module):
    """STANet-inspired U-Net with global temporal attention after encoding."""

    def __init__(self):
        super().__init__()
        self.encoder = SharedUNetEncoder()
        self.temporal_attention = SpatialTemporalAttention(512)

        self.fuse_bottleneck = FusionBlock(512)
        self.fuse4 = FusionBlock(256)
        self.fuse3 = FusionBlock(128)
        self.fuse2 = FusionBlock(64)
        self.fuse1 = FusionBlock(32)

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _match(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        features = self.encoder(torch.cat([a, b], dim=0))
        features_a = {key: value[: a.size(0)] for key, value in features.items()}
        features_b = {key: value[a.size(0) :] for key, value in features.items()}

        att_a, att_b = self.temporal_attention(features_a["bottleneck"], features_b["bottleneck"])
        x = self.fuse_bottleneck(att_a, att_b)

        s4 = self.fuse4(features_a["s4"], features_b["s4"])
        s3 = self.fuse3(features_a["s3"], features_b["s3"])
        s2 = self.fuse2(features_a["s2"], features_b["s2"])
        s1 = self.fuse1(features_a["s1"], features_b["s1"])

        x = self.dec4(torch.cat([self._match(self.up4(x), s4), s4], dim=1))
        x = self.dec3(torch.cat([self._match(self.up3(x), s3), s3], dim=1))
        x = self.dec2(torch.cat([self._match(self.up2(x), s2), s2], dim=1))
        x = self.dec1(torch.cat([self._match(self.up1(x), s1), s1], dim=1))
        return self.head(x)
