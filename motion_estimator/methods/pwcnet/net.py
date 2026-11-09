"""PWC-Net in one file: pyramid, warp, cost volume, decoder, context network.

A PyTorch rewrite of the TF1 tfoptflow fork in ``PWC/pwcNet_t1/pwcNet``,
kept to the same architecture (6 levels, search range 4, dense decoder,
dilated context network) but with three conventions simplified:

* **One input channel.** The old code copied greyscale into three channels
  because the pipeline was built for RGB; here the encoder takes greyscale.
* **Every level predicts flow in full-resolution pixels.** tfoptflow scaled
  flow by ``20 / 2**level`` for warping and resized the ground truth per
  level. Here a level's flow is warped with ``flow / 2**level`` and compared
  directly with the full-resolution displacement, so there is one unit
  everywhere and no magic 20.
* **Warping samples frame 2 at ``x + flow``.** The old warp used
  ``grid - flow`` with the channels read as ``(y, x)``, which the old code
  itself flagged as wrong (``core_warp.py:206``). Flow here is ``(dx, dy)``,
  x first, like everywhere else in this package.

Input height and width must be multiples of 64 (``2 ** levels``); the
estimator pads for you.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PWCNet", "warp", "cost_volume"]

PYRAMID_CHANNELS = (16, 32, 64, 96, 128, 196)  # levels 1..6
DECODER_CHANNELS = (128, 128, 96, 64, 32)


def conv(in_ch, out_ch, stride=1, dilation=1, activation=True):
    layer = nn.Conv2d(in_ch, out_ch, 3, stride, padding=dilation, dilation=dilation)
    if not activation:
        return layer
    return nn.Sequential(layer, nn.LeakyReLU(0.1, inplace=True))


def warp(image, flow):
    """Sample *image* at ``x + flow`` with bilinear interpolation.

    *flow* is ``(B, 2, H, W)`` in pixels of *image*, x-displacement first.
    Outside the image reads as zero.
    """
    _, _, height, width = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=flow.dtype),
        torch.arange(width, device=image.device, dtype=flow.dtype),
        indexing="ij",
    )
    x = xs[None] + flow[:, 0]
    y = ys[None] + flow[:, 1]
    grid = torch.stack(
        [2.0 * x / max(width - 1, 1) - 1.0, 2.0 * y / max(height - 1, 1) - 1.0],
        dim=-1,
    )
    return F.grid_sample(image, grid.to(image.dtype), mode="bilinear",
                         padding_mode="zeros", align_corners=True)


def cost_volume(first, second, search_range=4):
    """Correlation over a ``(2r+1)^2`` window, mean over channels.

    Plain torch -- a loop over 81 shifts -- so there is no custom CUDA op to
    build on Windows. Fast enough at the resolutions PWC-Net correlates at
    (1/4 and coarser).
    """
    _, _, height, width = first.shape
    padded = F.pad(second, [search_range] * 4)
    costs = []
    size = 2 * search_range + 1
    for dy in range(size):
        for dx in range(size):
            shifted = padded[:, :, dy: dy + height, dx: dx + width]
            costs.append((first * shifted).mean(dim=1, keepdim=True))
    return F.leaky_relu(torch.cat(costs, dim=1), 0.1)


class FeaturePyramid(nn.Module):
    """Six levels, each halving resolution; shared by both frames."""

    def __init__(self, in_ch=1):
        super().__init__()
        levels, previous = [], in_ch
        for channels in PYRAMID_CHANNELS:
            levels.append(nn.Sequential(
                conv(previous, channels, stride=2),
                conv(channels, channels),
                conv(channels, channels),
            ))
            previous = channels
        self.levels = nn.ModuleList(levels)

    def forward(self, image):
        features = []
        for level in self.levels:
            image = level(image)
            features.append(image)
        return features  # index 0 is level 1 (1/2), index 5 is level 6 (1/64)


class FlowDecoder(nn.Module):
    """DenseNet-style estimator: each conv sees every earlier output."""

    def __init__(self, in_ch):
        super().__init__()
        layers, channels = [], in_ch
        for out in DECODER_CHANNELS:
            layers.append(conv(channels, out))
            channels += out
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels
        self.predict = conv(channels, 2, activation=False)

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([layer(x), x], dim=1)
        return x, self.predict(x)


class ContextNetwork(nn.Module):
    """Dilated convolutions that refine the finest flow with a residual."""

    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_ch, 128, dilation=1),
            conv(128, 128, dilation=2),
            conv(128, 128, dilation=4),
            conv(128, 96, dilation=8),
            conv(96, 64, dilation=16),
            conv(64, 32, dilation=1),
            conv(32, 2, activation=False),
        )

    def forward(self, x):
        return self.net(x)


class PWCNet(nn.Module):
    """Two frames in, dense flow out.

    ``forward`` returns a dict:

    * ``flow`` -- ``(B, 2, H, W)`` at input resolution, in pixels.
    * ``pyramid`` -- list of ``(level, flow)`` from coarsest to finest, each
      flow at that level's resolution but in *full-resolution* pixels. The
      multiscale loss reads this.
    """

    def __init__(self, search_range=4, top_level=6, bottom_level=2, in_ch=1):
        super().__init__()
        self.search_range = search_range
        self.top_level = top_level
        self.bottom_level = bottom_level
        self.pyramid = FeaturePyramid(in_ch)

        n_corr = (2 * search_range + 1) ** 2
        self.decoders = nn.ModuleDict()
        self.upsample_feat = nn.ModuleDict()
        for level in range(top_level, bottom_level - 1, -1):
            in_ch = n_corr if level == top_level else (
                n_corr + PYRAMID_CHANNELS[level - 1] + 4
            )
            decoder = FlowDecoder(in_ch)
            self.decoders[str(level)] = decoder
            if level > bottom_level:
                self.upsample_feat[str(level)] = nn.ConvTranspose2d(
                    decoder.out_channels, 2, 4, 2, 1
                )
        self.context = ContextNetwork(self.decoders[str(bottom_level)].out_channels)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    @staticmethod
    def normalise(image):
        """Zero mean, unit spread per image, so gain differences do not matter."""
        mean = image.mean(dim=(1, 2, 3), keepdim=True)
        std = image.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-3)
        return (image - mean) / std

    def forward(self, image0, image1):
        height, width = image0.shape[2:]
        multiple = 2 ** self.top_level
        if height % multiple or width % multiple:
            raise ValueError(f"input {height}x{width} is not a multiple of {multiple}")

        features = self.pyramid(torch.cat([self.normalise(image0),
                                           self.normalise(image1)], dim=0))
        batch = image0.shape[0]
        pyramid, flow, up_feat, hidden = [], None, None, None

        for level in range(self.top_level, self.bottom_level - 1, -1):
            first, second = features[level - 1].split(batch)
            scale = 2.0 ** level
            if flow is None:
                corr = cost_volume(first, second, self.search_range)
                x = corr
            else:
                up_flow = F.interpolate(flow, scale_factor=2, mode="bilinear",
                                        align_corners=False)
                warped = warp(second, up_flow / scale)
                corr = cost_volume(first, warped, self.search_range)
                x = torch.cat([corr, first, up_flow / scale, up_feat], dim=1)

            hidden, delta = self.decoders[str(level)](x)
            # Decoder output is in this level's pixels; store full-res pixels.
            flow = delta * scale if flow is None else up_flow + delta * scale
            if level == self.bottom_level:
                flow = flow + self.context(hidden) * scale
            else:
                up_feat = self.upsample_feat[str(level)](hidden)
            pyramid.append((level, flow))

        full = F.interpolate(flow, size=(height, width), mode="bilinear",
                             align_corners=False)
        return {"flow": full, "pyramid": pyramid}


if __name__ == "__main__":
    # Smoke test: shapes and memory at the training crop size.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PWCNet().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    a = torch.rand(8, 1, 320, 320, device=device)
    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
        out = model(a, a)
        out["flow"].float().abs().mean().backward()
    print("flow", tuple(out["flow"].shape),
          [(lvl, tuple(f.shape[2:])) for lvl, f in out["pyramid"]])
    if device == "cuda":
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
