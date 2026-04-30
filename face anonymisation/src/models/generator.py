import torch
from torch import nn


class UNetAnonymizer(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        noise_channels: int = 1,
        bottleneck_dropout: float = 0.2,
        bottleneck_blocks: int = 0,
        residual_output: bool = False,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.noise_channels = int(noise_channels)
        self.residual_output = bool(residual_output)
        self.residual_scale = float(residual_scale)

        total_in = in_channels + self.noise_channels
        self.down1 = _down_block(total_in, base_channels, use_norm=False)
        self.down2 = _down_block(base_channels, base_channels * 2)
        self.down3 = _down_block(base_channels * 2, base_channels * 4)
        self.down4 = _down_block(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=bottleneck_dropout),
        )
        self.extra_bottleneck = (
            nn.Sequential(*[_ResidualBlock(base_channels * 8) for _ in range(int(bottleneck_blocks))])
            if int(bottleneck_blocks) > 0
            else nn.Identity()
        )

        self.up1 = _up_block(base_channels * 8, base_channels * 4)
        self.up2 = _up_block(base_channels * 8, base_channels * 2)
        self.up3 = _up_block(base_channels * 4, base_channels)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        if self.residual_output:
            self.residual_head = nn.ConvTranspose2d(
                base_channels * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(self, image: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.zeros(
                (image.size(0), self.noise_channels, image.size(2), image.size(3)),
                device=image.device,
                dtype=image.dtype,
            )
        x = torch.cat([image, noise], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bottleneck = self.extra_bottleneck(self.bottleneck(d4))

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        up_input = torch.cat([u3, d1], dim=1)
        if self.residual_output:
            delta = torch.tanh(self.residual_head(up_input))
            if delta.shape[-2:] != image.shape[-2:]:
                image = torch.nn.functional.interpolate(
                    image,
                    size=delta.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            out = torch.clamp(image + delta * self.residual_scale, min=-1.0, max=1.0)
        else:
            out = self.up4(up_input)
        return out


def _down_block(in_channels: int, out_channels: int, use_norm: bool = True) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm),
    ]
    if use_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))
