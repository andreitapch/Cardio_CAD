"""Score one pair of feature maps against known point motion.

For each query the feature at its cell in frame 0 is compared with frame 1
cells inside a ``(2R+1)^2`` window, R = 4 cells like PWC-Net's cost volume.
Returned per query, so the caller can pool medians over many pairs:

* ``hit``      -- the best cell is the cell the true target lies in
* ``within1``  -- the best cell is at most one cell off
* ``nn_epe``   -- |best-cell displacement - true displacement|, px
* ``oracle``   -- the same error for the *true* cell: the quantisation floor
* ``soft_epe`` -- sub-cell estimate from a softmax over the 3x3 around the best cell, px
* ``gain``     -- soft estimate projected on the true motion / |true|, where |true| > 1 px
* ``margin``   -- similarity at the true cell minus the best cell two or more away
* ``global1``  -- the best cell over the whole map is within one cell of the truth
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

R = 4
TAU = 0.1


def score(f0, f1, points0, points1, stride, centre=False):
    """*f0*, *f1* ``(C, h, w)``; points ``(N, 2)`` 0-based crop pixels. Returns dict of ``(N',)`` tensors.

    *centre* subtracts each map's spatial mean first, removing a component shared by every cell.
    """
    if centre:
        f0 = f0 - f0.mean(dim=(1, 2), keepdim=True)
        f1 = f1 - f1.mean(dim=(1, 2), keepdim=True)
    c, h, w = f0.shape
    device = f0.device
    p0 = torch.as_tensor(points0, device=device, dtype=torch.float32)
    p1 = torch.as_tensor(points1, device=device, dtype=torch.float32)
    cell0 = torch.floor((p0 + 0.5) / stride).long()
    cell1 = torch.floor((p1 + 0.5) / stride).long()
    keep = ((cell0 >= 0) & (cell0[:, :1] < w) & (cell0[:, 1:] < h)
            & (cell1 >= 0) & (cell1[:, :1] < w) & (cell1[:, 1:] < h)).all(1)
    p0, p1, cell0, cell1 = p0[keep], p1[keep], cell0[keep], cell1[keep]
    if p0.shape[0] == 0:
        return {}
    true = p1 - p0
    true_off = cell1 - cell0                                              # (N, 2) cells, (dx, dy)

    f0n = F.normalize(f0, dim=0)
    f1n = F.normalize(f1, dim=0)
    q = f0n[:, cell0[:, 1], cell0[:, 0]].T                                # (N, C)

    size = 2 * R + 1
    padded = F.pad(f1n, [R] * 4)
    valid = F.pad(torch.ones(1, h, w, device=device), [R] * 4)
    windows = F.unfold(padded[None], size)[0].view(c, size * size, h, w)  # (C, k, h, w)
    vwin = F.unfold(valid[None], size)[0].view(size * size, h, w)
    win = windows[:, :, cell0[:, 1], cell0[:, 0]].permute(2, 0, 1)        # (N, C, k)
    vmask = vwin[:, cell0[:, 1], cell0[:, 0]].T > 0.5                     # (N, k)
    sims = torch.einsum("nc,nck->nk", q, win).masked_fill(~vmask, -2.0)  # (N, k)

    offsets = torch.stack(torch.meshgrid(torch.arange(-R, R + 1, device=device),
                                         torch.arange(-R, R + 1, device=device), indexing="ij"), -1)
    offsets = offsets.view(-1, 2).flip(-1).float()                        # (k, 2) as (dx, dy)
    best = sims.argmax(1)
    best_off = offsets[best]
    cheb = (best_off - true_off.float()).abs().max(1).values
    in_window = (true_off.abs() <= R).all(1)
    true_idx = ((true_off[:, 1] + R) * size + (true_off[:, 0] + R)).clamp(0, size * size - 1)
    sim_true = sims.gather(1, true_idx[:, None])[:, 0]
    far = (offsets[None] - true_off[:, None].float()).abs().max(-1).values >= 2
    sim_far = sims.masked_fill(~far, -2.0).max(1).values

    # Sub-cell: softmax over the 3x3 around the best cell.
    neigh = best_off[:, None] + offsets[None, (offsets.abs().max(1).values <= 1)]   # (N, 9, 2)
    nidx = ((neigh[..., 1] + R) * size + (neigh[..., 0] + R)).long()
    inside = (neigh.abs() <= R).all(-1)
    nsims = sims.gather(1, nidx.clamp(0, size * size - 1)).masked_fill(~inside, -1e4)
    weights = torch.softmax(nsims / TAU, 1)
    soft_off = (weights[..., None] * neigh).sum(1)                        # (N, 2) cells
    soft_px = soft_off * stride
    true_norm = true.norm(dim=1)
    moving = true_norm > 1.0
    gain = (soft_px * true).sum(1) / true_norm.clamp_min(1e-6) ** 2

    # Global nearest neighbour over the whole map.
    allsims = q @ f1n.view(c, -1)                                         # (N, h*w)
    g = allsims.argmax(1)
    g_off = torch.stack([g % w, g // w], 1) - cell0
    global1 = (g_off - true_off).abs().max(1).values <= 1

    return {
        "hit": (cheb == 0).float(),
        "within1": (cheb <= 1).float(),
        "nn_epe": (best_off * stride - true).norm(dim=1),
        "oracle": (true_off.float() * stride - true).norm(dim=1),
        "soft_epe": (soft_px - true).norm(dim=1),
        "gain": torch.where(moving, gain, torch.nan),
        "margin": torch.where(in_window, sim_true - sim_far, torch.nan),
        "global1": global1.float(),
        "zero": true_norm,
    }
