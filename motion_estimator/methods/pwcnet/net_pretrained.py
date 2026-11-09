"""PWC-Net with FlyingChairs-pretrained weights (the IRR implementation).

Vendored from Hur & Roth's IRR repository (``models/pwcnet.py`` and
``models/pwc_modules.py``, Apache License 2.0, D:\\PHD\\Development\\Python\\irr),
with two changes: no hard-coded ``.cuda()`` calls, and a wrapper that speaks
this package's interface -- greyscale in, flow in full-resolution pixels out,
plus the per-level pyramid that :func:`train_paper.dense_loss` reads.

Same architecture family as :mod:`.net` (6-level pyramid, search range 4,
DenseNet decoder, dilated context network, output at 1/4 resolution); the
pyramid has two convolutions per level instead of three, and flows are carried
in IRR's ``x 0.05`` units internally.

**Why pretrained.** Trained from scratch on the simulated pairs, :mod:`.net`
sat at the zero-flow error for 5,000 steps (val EPE 0.977 px; predicting no
motion scores ~1.0), and widening the pairs to 1-8 frames apart did not move it
either. These weights, with no echo data at all, already scored 0.53 px against
1.13 px for zero flow on synthetic validation pairs, and 0.97 vs 1.32 px on in
vivo ones.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

IRR_CHECKPOINT = Path(r"D:\PHD\Development\Python\irr\saved_check_point\pwcnet\PWCNet\checkpoint_best.ckpt")
DIV_FLOW = 0.05


def _conv(ch_in, ch_out, kernel=3, stride=1, dilation=1, relu=True):
    layers = [nn.Conv2d(ch_in, ch_out, kernel, stride, ((kernel - 1) * dilation) // 2, dilation)]
    if relu:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)


def _cost_volume(first, second, max_disp=4):
    _, _, height, width = first.shape
    padded = F.pad(second, [max_disp] * 4)
    shifts = 2 * max_disp + 1
    return torch.cat([(first * padded[:, :, i:i + height, j:j + width]).mean(1, keepdim=True)
                      for i in range(shifts) for j in range(shifts)], dim=1)


def _warp(x, flow, height_im, width_im):
    """IRR's warping layer: *flow* in x 0.05 full-resolution units."""
    _, _, h, w = x.shape
    grid_x = torch.linspace(-1.0, 1.0, w, device=x.device).view(1, 1, w).expand(x.size(0), h, w)
    grid_y = torch.linspace(-1.0, 1.0, h, device=x.device).view(1, h, 1).expand(x.size(0), h, w)
    grid = torch.stack([grid_x + flow[:, 0].float() * 2 / max(width_im - 1, 1) / DIV_FLOW,
                        grid_y + flow[:, 1].float() * 2 / max(height_im - 1, 1) / DIV_FLOW], dim=-1)
    warped = F.grid_sample(x.float(), grid, align_corners=True)
    inside = F.grid_sample(torch.ones_like(x, dtype=torch.float32), grid, align_corners=True) >= 1.0
    return (warped * inside).to(x.dtype)


class _FeatureExtractor(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.convs = nn.ModuleList(nn.Sequential(_conv(a, b, stride=2), _conv(b, b))
                                   for a, b in zip(chs[:-1], chs[1:]))

    def forward(self, x):
        pyramid = []
        for conv in self.convs:
            x = conv(x)
            pyramid.append(x)
        return pyramid[::-1]


class _FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.conv1 = _conv(ch_in, 128)
        self.conv2 = _conv(ch_in + 128, 128)
        self.conv3 = _conv(ch_in + 256, 96)
        self.conv4 = _conv(ch_in + 352, 64)
        self.conv5 = _conv(ch_in + 416, 32)
        self.conv_last = _conv(ch_in + 448, 2, relu=False)

    def forward(self, x):
        for conv in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            x = torch.cat([conv(x), x], dim=1)
        return x, self.conv_last(x)


class _ContextNetwork(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.convs = nn.Sequential(
            _conv(ch_in, 128, dilation=1), _conv(128, 128, dilation=2), _conv(128, 128, dilation=4),
            _conv(128, 96, dilation=8), _conv(96, 64, dilation=16), _conv(64, 32, dilation=1),
            _conv(32, 2, relu=False))

    def forward(self, x):
        return self.convs(x)


class IRRPWCNet(nn.Module):
    """IRR's ``PWCNet``; parameter names match its checkpoints after dropping ``_model.``."""

    def __init__(self):
        super().__init__()
        chs = [3, 16, 32, 64, 96, 128, 196]
        self.output_level = 4
        self.feature_pyramid_extractor = _FeatureExtractor(chs)
        corr = (2 * 4 + 1) ** 2
        self.flow_estimators = nn.ModuleList(
            _FlowEstimatorDense(corr if level == 0 else corr + ch + 2)
            for level, ch in enumerate(chs[::-1][:self.output_level + 1]))
        self.context_networks = _ContextNetwork(corr + 32 + 2 + 448 + 2)

    def forward(self, image0, image1):
        """Flows coarsest to finest (1/64 ... 1/4), in x 0.05 full-resolution pixels."""
        height, width = image0.shape[2:]
        pyramid0 = self.feature_pyramid_extractor(image0) + [image0]
        pyramid1 = self.feature_pyramid_extractor(image1) + [image1]
        flows, flow = [], None
        for level, (x1, x2) in enumerate(zip(pyramid0, pyramid1)):
            if level == 0:
                warped = x2
            else:
                flow = F.interpolate(flow, x1.shape[2:], mode="bilinear", align_corners=True)
                warped = _warp(x2, flow, height, width)
            corr = F.leaky_relu(_cost_volume(x1, warped), 0.1)
            if level == 0:
                hidden, flow = self.flow_estimators[level](corr)
            else:
                hidden, flow = self.flow_estimators[level](torch.cat([corr, x1, flow], dim=1))
            if level == self.output_level:
                flow = flow + self.context_networks(torch.cat([hidden, flow], dim=1))
                flows.append(flow)
                break
            flows.append(flow)
        return flows


class PretrainedPWCNet(nn.Module):
    """Greyscale in, ``{"flow", "pyramid"}`` out, like :class:`.net.PWCNet`.

    Images are fed as IRR was trained: ``[0, 1]``, the grey channel repeated
    three times. Input sides must be multiples of 64.
    """

    def __init__(self, checkpoint: str | Path | None = IRR_CHECKPOINT):
        super().__init__()
        self.net = IRRPWCNet()
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            self.net.load_state_dict({k.removeprefix("_model."): v for k, v in state.items()})

    def forward(self, image0, image1):
        height, width = image0.shape[2:]
        if height % 64 or width % 64:
            raise ValueError(f"input {height}x{width} is not a multiple of 64")
        flows = self.net(image0.repeat(1, 3, 1, 1), image1.repeat(1, 3, 1, 1))
        pyramid = [(6 - i, flow / DIV_FLOW) for i, flow in enumerate(flows)]  # levels 6..2, full-res px
        full = F.interpolate(pyramid[-1][1], size=(height, width), mode="bilinear", align_corners=True)
        return {"flow": full, "pyramid": pyramid}
