"""The three losses: sparse points, photometric, smoothness.

* :func:`point_loss` -- supervised. Both the simulated scatterers and MITT's
  contour are *sparse*, so the predicted flow is sampled at the start points
  and compared with the known displacement. No dense ground truth is needed
  or invented.
* :func:`photometric_loss` + :func:`smoothness_loss` -- unsupervised, for in
  vivo fine-tuning without trusting MITT. Frame 2 is warped back by the
  predicted flow and must look like frame 1 inside the sector.

Level weights put the most weight on the finest level, as the old echo runs
did (``0.015 ... 0.5``, "Myocardium motion paper" set).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .net import warp

__all__ = ["LEVEL_WEIGHTS", "sample_flow", "point_loss", "photometric_loss",
           "smoothness_loss"]

LEVEL_WEIGHTS = {6: 0.03, 5: 0.06, 4: 0.12, 3: 0.25, 2: 0.5}


def sample_flow(flow, points, scale=1.0):
    """Bilinear sample of ``(B, 2, h, w)`` flow at 0-based full-res *points*.

    *points* is ``(B, N, 2)`` as ``(x, y)`` in full-resolution pixels; *scale*
    is how much smaller *flow* is than full resolution. Returns ``(B, N, 2)``.
    """
    _, _, height, width = flow.shape
    # Level pixel i covers full-res pixels [i*s, (i+1)*s), so full-res x maps
    # to level coordinate (x + 0.5) / s - 0.5; with align_corners=False that
    # normalises to 2 * (x + 0.5) / (s * size) - 1.
    x = 2.0 * (points[..., 0] + 0.5) / (scale * width) - 1.0
    y = 2.0 * (points[..., 1] + 0.5) / (scale * height) - 1.0
    grid = torch.stack([x, y], dim=-1)
    sampled = F.grid_sample(flow.float(), grid[:, :, None].float(), mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled[..., 0].permute(0, 2, 1)


def point_loss(output, points0, points1, valid, weights=LEVEL_WEIGHTS):
    """Multiscale end-point error at sparse points.

    Returns ``(loss, epe)`` where *epe* is the full-resolution mean end-point
    error in pixels, for logging.
    """
    target = points1 - points0
    mask = valid.float()
    count = mask.sum().clamp_min(1.0)
    loss = 0.0
    for level, flow in output["pyramid"]:
        predicted = sample_flow(flow, points0, 2.0 ** level)
        error = torch.linalg.norm(predicted - target, dim=-1)
        loss = loss + weights.get(level, 0.0) * (error * mask).sum() / count
    predicted = sample_flow(output["flow"], points0, 1.0)
    epe = (torch.linalg.norm(predicted - target, dim=-1) * mask).sum() / count
    return loss, epe.detach()


def _charbonnier(x, eps=1e-3, alpha=0.45):
    return (x * x + eps * eps) ** alpha


def photometric_loss(image0, image1, flow, mask=None):
    """Charbonnier difference between frame 1 and frame 2 warped back.

    Only counted inside the sector (*mask*) and where the warp stayed inside
    frame 2 -- outside, the warped image is zero padding and the loss would
    reward flow that points off the image.
    """
    warped = warp(image1.float(), flow.float())
    inside = warp(torch.ones_like(image1, dtype=torch.float32), flow.float()) > 0.99
    weight = inside.float()
    if mask is not None:
        weight = weight * mask.float()
    diff = _charbonnier(image0.float() - warped)
    return (diff * weight).sum() / weight.sum().clamp_min(1.0)


def smoothness_loss(image, flow, edge_weight=10.0):
    """Edge-aware first-order smoothness; flow in pixels is divided by size.

    Dividing by the image size keeps the weight meaningful across crop sizes.
    """
    flow = flow.float() / image.shape[-1]
    image = image.float()
    dx_flow = flow[..., :, 1:] - flow[..., :, :-1]
    dy_flow = flow[..., 1:, :] - flow[..., :-1, :]
    dx_img = (image[..., :, 1:] - image[..., :, :-1]).abs().mean(1, keepdim=True)
    dy_img = (image[..., 1:, :] - image[..., :-1, :]).abs().mean(1, keepdim=True)
    wx = torch.exp(-edge_weight * dx_img)
    wy = torch.exp(-edge_weight * dy_img)
    return (dx_flow.abs() * wx).mean() + (dy_flow.abs() * wy).mean()
