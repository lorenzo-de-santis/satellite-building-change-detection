import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """Shallow full-resolution CNN baseline with no pooling or decoder.

    The model returns raw logits. The dilations follow a small HDC-style pattern
    to avoid the gridding artifacts caused by adjacent rates such as 2 and 4.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(6, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 48, dilation=2),
            ConvBlock(48, 48, dilation=3),
            ConvBlock(48, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.out_channels == 1 and module.kernel_size == (1, 1):
                    nn.init.xavier_normal_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)
        return self.net(x)
