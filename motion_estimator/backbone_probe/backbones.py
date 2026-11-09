"""Frozen feature extractors, one builder per backbone.

Every extractor takes greyscale ``(B, 1, S, S)`` in ``[0, 1]`` and returns
``{stride: (B, C, h, w)}`` float32 maps. The stride is in input pixels and may
be fractional (a ViT run at 518 on a 512 crop). Nothing here has gradients.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
PAPER_RUNS = ARTIFACTS / "paper-runs"
CNN_STRIDES = (4, 8, 16, 32)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class Extractor:
    def __init__(self, name, fn, params, note=""):
        self.name, self.fn, self.params, self.note = name, fn, params, note

    @torch.inference_mode()
    def __call__(self, images, amp=True):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and images.is_cuda):
            maps = self.fn(images)
        return {stride: m.float() for stride, m in maps.items()}


def _rgb(images, mean, std):
    x = images.repeat(1, 3, 1, 1)
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _count(module):
    return sum(p.numel() for p in module.parameters())


# -- baselines --------------------------------------------------------------

def pixels(device):
    """Zero-mean raw patches at each stride: block matching with no learning."""
    def fn(images):
        out = {}
        for s in CNN_STRIDES:
            patches = F.unfold(images, s, stride=s)                      # (B, s*s, h*w)
            patches = patches - patches.mean(1, keepdim=True)
            h = w = images.shape[-1] // s
            out[s] = patches.view(images.shape[0], s * s, h, w)
        return out
    return Extractor("pixels", fn, 0, "raw zero-mean patches")


def _pwc_scratch(name, checkpoint, device, note):
    from ..methods.pwcnet.net import PWCNet

    model = PWCNet()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    model = model.eval().to(device)

    def fn(images):
        maps = model.pyramid(PWCNet.normalise(images))                  # strides 2..64
        return {2 ** (i + 1): m for i, m in enumerate(maps) if 2 ** (i + 1) in CNN_STRIDES}
    return Extractor(name, fn, _count(model.pyramid), note)


def _pwc_irr(name, checkpoint, device, note):
    from ..methods.pwcnet.net_pretrained import IRR_CHECKPOINT, PretrainedPWCNet

    if checkpoint is None:
        model = PretrainedPWCNet(IRR_CHECKPOINT)
    else:
        model = PretrainedPWCNet(None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
    model = model.eval().to(device)

    def fn(images):
        maps = model.net.feature_pyramid_extractor(images.repeat(1, 3, 1, 1))  # coarsest first, 64..2
        return {2 ** (6 - i): m for i, m in enumerate(maps) if 2 ** (6 - i) in CNN_STRIDES}
    return Extractor(name, fn, _count(model.net.feature_pyramid_extractor), note)


# -- foundation models ------------------------------------------------------

def _timm_cnn(name, model_id, device, note):
    import timm

    model = timm.create_model(model_id, pretrained=True, features_only=True,
                              out_indices=(0, 1, 2, 3)).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    strides = model.feature_info.reduction()

    def fn(images):
        return dict(zip(strides, model(_rgb(images, cfg["mean"], cfg["std"]))))
    return Extractor(name, fn, _count(model), note)


def _timm_vit(name, model_id, size, blocks, device, note):
    import timm

    model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=size,
                              dynamic_img_size=True).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)

    def fn(images):
        x = _rgb(images, cfg["mean"], cfg["std"])
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        maps = model.forward_intermediates(x, indices=list(blocks), norm=False,
                                           output_fmt="NCHW", intermediates_only=True)
        stride = images.shape[-1] / maps[0].shape[-1]
        out = {(stride, f"b{block + 1}"): m for block, m in zip(blocks, maps)}  # one level per block
        for factor in (2, 4):  # the last block pooled to coarser levels, where the stride stays integral
            if float(stride * factor).is_integer() and images.shape[-1] % int(stride * factor) == 0:
                out[(stride * factor, f"b{blocks[-1] + 1}p{factor}")] = F.avg_pool2d(maps[-1], factor)
        return out
    return Extractor(name, fn, _count(model), note + f"; blocks {list(blocks)} at {size}")


def echoclip(device):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:mkaichristensen/echo-clip", precision="fp16", device=device)
    trunk = model.visual.trunk.eval()                                   # timm ConvNeXt-Base

    def fn(images):
        x = trunk.stem(_rgb(images, CLIP_MEAN, CLIP_STD).half())
        out = {}
        for stride, stage in zip(CNN_STRIDES, trunk.stages):
            x = stage(x)
            out[stride] = x
        return out
    return Extractor("echoclip_convnext_b", fn, _count(trunk), "EchoCLIP image trunk, echo domain")


BUILDERS = {
    "pixels": lambda d: pixels(d),
    "pwc_random": lambda d: _pwc_scratch("pwc_random", None, d, "untrained PWC-Net pyramid"),
    "pwc_scratch_ms47": lambda d: _pwc_scratch(
        "pwc_scratch_ms47", PAPER_RUNS / "MS47_IV0" / "single" / "best.pt", d,
        "PWC-Net pyramid trained from scratch on synthetic only"),
    "pwc_scratch_ms47iv85": lambda d: _pwc_scratch(
        "pwc_scratch_ms47iv85", PAPER_RUNS / "MS47_IV85" / "fold0" / "best.pt", d,
        "PWC-Net pyramid trained from scratch, synthetic + MITT in vivo (fold 0)"),
    "pwc_chairs": lambda d: _pwc_irr("pwc_chairs", None, d, "FlyingChairs weights, no echo"),
    "pwc_chairs_ms47": lambda d: _pwc_irr(
        "pwc_chairs_ms47", PAPER_RUNS / "MS47_IV0_pretrained150k" / "single" / "best.pt", d,
        "FlyingChairs then synthetic only"),
    "pwc_chairs_ms47iv85": lambda d: _pwc_irr(
        "pwc_chairs_ms47iv85", PAPER_RUNS / "MS47_IV85_pretrained150k" / "fold0" / "best.pt", d,
        "FlyingChairs then synthetic + MITT in vivo (fold 0): the current best"),
    "dinov3_convnext_t": lambda d: _timm_cnn(
        "dinov3_convnext_t", "convnext_tiny.dinov3_lvd1689m", d, "DINOv3 distilled ConvNeXt-Tiny"),
    "dinov3_vit_s16": lambda d: _timm_vit(
        "dinov3_vit_s16", "vit_small_patch16_dinov3.lvd1689m", 512, (8, 11), d, "DINOv3 ViT-S/16"),
    "dinov2_vit_s14": lambda d: _timm_vit(
        "dinov2_vit_s14", "vit_small_patch14_reg4_dinov2.lvd142m", 518, (8, 11), d, "DINOv2 ViT-S/14 reg"),
    "echoclip": lambda d: echoclip(d),
}
