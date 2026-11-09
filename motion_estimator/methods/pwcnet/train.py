"""Train PWC-Net on one fold: simulated pretraining or in-vivo fine-tuning.

    # 1. simulated pretraining, fold k (point loss on exact scatterers)
    python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold 0 --cohort simulated

    # 2. in-vivo fine-tuning from that fold's model
    python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold 0 --cohort in_vivo --mode unsup
    python -m motion_estimator.methods.pwcnet.train --run-dir <cv> --fold 0 --cohort in_vivo --mode mitt

*run-dir* is the cross-validation directory holding ``folds.json``; each
training writes into its own subfolder:

    <cv>/sim_fold0/            best.pt, last.pt, history.json, config.json
    <cv>/invivo_fold0_unsup/
    <cv>/invivo_fold0_mitt/

``best.pt`` is chosen on the fold's **validation** patients, never its test
patients. For ``unsup`` the selection criterion is the photometric loss, so
MITT does not leak into the unsupervised model at all; the MITT end-point
error on validation is still logged beside it for reference.

Defaults fit a 6 GB RTX 2060: 320 crops, batch 8, fp16 autocast (not bf16,
which fails in cuDNN on Turing).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ...runs import write_config
from .data import SIM_CACHE, InVivoPairDataset, list_invivo_cached, load_folds
from .losses import photometric_loss, point_loss, smoothness_loss
from .net import PWCNet

DEFAULTS = {
    # cohort: (steps, lr, min_spacing, max_spacing)
    "simulated": (30000, 1e-4, 1, 4),
    "in_vivo": (3000, 2e-5, 1, 3),
}


def output_dir(run_dir, cohort, fold, mode):
    name = f"sim_fold{fold}" if cohort == "simulated" else f"invivo_fold{fold}_{mode}"
    return Path(run_dir) / name


def build_datasets(args, fold):
    if args.cohort == "simulated":
        from ...loaders.train_loader import SimPairDataset, list_cached

        clips = list_cached(args.sim_cache)
        by_group = lambda groups: [c for c in clips if c.group_id in set(groups)]
        common = dict(size=args.size, min_spacing=args.min_spacing,
                      max_spacing=args.max_spacing, n_points=args.n_points,
                      clip_run=args.batch_size)
        train = SimPairDataset(by_group(fold["train"]),
                               epoch_length=args.steps * args.batch_size,
                               seed=args.seed, **common)
        val = SimPairDataset(by_group(fold["val"]),
                             epoch_length=args.val_batches * args.batch_size,
                             seed=args.seed + 1, augment=False, **common)
        return train, val

    items = list_invivo_cached()
    paths = lambda groups: [i["path"] for i in items if i["group_id"] in set(groups)]
    common = dict(size=args.size, min_spacing=args.min_spacing,
                  max_spacing=args.max_spacing, n_points=min(args.n_points, 128))
    train = InVivoPairDataset(paths(fold["train"]),
                              epoch_length=args.steps * args.batch_size,
                              seed=args.seed, **common)
    val = InVivoPairDataset(paths(fold["val"]),
                            epoch_length=args.val_batches * args.batch_size,
                            seed=args.seed + 1, augment=False, **common)
    return train, val


def compute_loss(model, batch, device, args):
    image0 = batch["image0"].to(device, non_blocking=True)
    image1 = batch["image1"].to(device, non_blocking=True)
    output = model(image0, image1)
    points = [batch[k].to(device) for k in ("points0", "points1", "valid")]
    supervised, epe = point_loss(output, *points)
    parts = {"epe": float(epe)}

    if args.mode == "unsup":
        sector = batch["sector"].to(device) if "sector" in batch else None
        photo = photometric_loss(image0, image1, output["flow"], sector)
        smooth = smoothness_loss(image0, output["flow"])
        parts.update(photo=float(photo), smooth=float(smooth))
        return photo + args.smooth_weight * smooth, parts
    return supervised, parts


@torch.no_grad()
def validate(model, loader, device, args):
    """Mean selection loss and mean end-point error over the validation pairs."""
    model.eval()
    losses, epes = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
            loss, parts = compute_loss(model, batch, device, args)
        losses.append(float(loss))
        epes.append(parts["epe"])
    model.train()
    return float(np.nanmean(losses)), float(np.nanmean(epes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="CV directory with folds.json")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--cohort", choices=("simulated", "in_vivo"), required=True)
    parser.add_argument("--mode", choices=("points", "unsup", "mitt"), default=None,
                        help="simulated: points (default); in_vivo: unsup or mitt")
    parser.add_argument("--init", default=None,
                        help="checkpoint to start from; in_vivo defaults to "
                             "<run-dir>/sim_fold<k>/best.pt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=320, help="crop, multiple of 64")
    parser.add_argument("--min-spacing", type=int, default=None)
    parser.add_argument("--max-spacing", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--smooth-weight", type=float, default=10.0)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sim-cache", default=str(SIM_CACHE))
    args = parser.parse_args(argv)

    if args.size % 64:
        parser.error("--size must be a multiple of 64")
    if args.mode is None:
        args.mode = "points" if args.cohort == "simulated" else "unsup"
    if args.cohort == "in_vivo" and args.mode == "points":
        args.mode = "mitt"  # points on in-vivo data are MITT's
    steps, lr, low, high = DEFAULTS[args.cohort]
    args.steps = args.steps or steps
    args.lr = args.lr or lr
    args.min_spacing = args.min_spacing or low
    args.max_spacing = args.max_spacing or high

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = (not args.no_amp) and device.type == "cuda"
    torch.manual_seed(args.seed)

    fold = load_folds(args.run_dir)[args.cohort][args.fold]
    out = output_dir(args.run_dir, args.cohort, args.fold, args.mode)
    out.mkdir(parents=True, exist_ok=True)

    if args.cohort == "in_vivo" and args.init is None:
        args.init = str(Path(args.run_dir) / f"sim_fold{args.fold}" / "best.pt")
    model = PWCNet().to(device)
    if args.init:
        state = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model", state))
        print(f"initialised from {args.init}")

    train_set, val_set = build_datasets(args, fold)
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=0, drop_last=True
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-4)
    milestones = [int(args.steps * f) for f in (0.5, 0.75, 0.9)]
    schedule = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    write_config(out, {"stage": f"train_{args.cohort}", **vars(args),
                       "fold_patients": fold, "run_dir": str(out)})
    print(f"{args.cohort} fold {args.fold} ({args.mode}): "
          f"{len(fold['train'])} train / {len(fold['val'])} val / "
          f"{len(fold['test'])} test patients -> {out}")

    history, best, started = [], float("inf"), time.time()
    model.train()
    for step, batch in enumerate(loader, 1):
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
            loss, parts = compute_loss(model, batch, device, args)
        if not torch.isfinite(loss):
            print(f"  step {step}: non-finite loss, skipped", file=sys.stderr)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        before = scaler.get_scale()
        scaler.step(optimiser)
        scaler.update()
        if scaler.get_scale() >= before:  # only when the optimiser really stepped
            schedule.step()

        if step % 50 == 0:
            rate = (time.time() - started) / step
            extra = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
            print(f"[{step}/{args.steps}] loss {float(loss):.4f} {extra} "
                  f"lr {schedule.get_last_lr()[0]:.1e} {rate:.2f}s/step "
                  f"eta {(args.steps - step) * rate / 60:.0f}min")
        if step % args.val_every == 0 or step == args.steps:
            value, epe = validate(model, val_loader, device, args)
            history.append({"step": step, "val_loss": value, "val_epe": epe})
            marker = ""
            if value < best:
                best = value
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "step": step}, out / "best.pt")
                marker = "  <- best, saved"
            print(f"  validation loss {value:.4f}  epe {epe:.3f} px{marker}")
            (out / "history.json").write_text(
                json.dumps({"history": history}, indent=2), encoding="utf-8")
        if step >= args.steps:
            break

    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")
    print(f"best validation {best:.4f}; {(time.time() - started) / 60:.0f} min; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
