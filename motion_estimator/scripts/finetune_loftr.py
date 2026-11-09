"""Fine-tune LoFTR on the simulated cohort's exact ground-truth motion.

    python -m motion_estimator.scripts.finetune_loftr --steps 2000
    python -m motion_estimator.scripts.finetune_loftr --unfreeze-backbone --size 256

Runs the kornia LoFTR modules by hand rather than calling ``forward``, because
``forward`` returns matched keypoints and the losses need the coarse assignment
matrix and the fine offsets, which it does not expose.

Defaults are set for 6 GB (an RTX 2060), where the official recipe does not
fit: 320 crops, batch 1, backbone frozen, fp16 autocast. **fp16 and not bf16** --
on Turing, bf16 raises inside cuDNN; see ``docs/environment-notes.md``.

Freezing the backbone by default is not timidity: there are ~46 patients behind
the training set, and the trunk is the part most likely to overfit them.
``--unfreeze-backbone`` is there to be measured against, not assumed better.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from motion_estimator.methods.loftr.losses import coarse_focal_loss, fine_l2_loss
from motion_estimator.loaders.train_loader import (
    SimPairDataset,
    list_cached,
    split_by_patient,
)
from motion_estimator.runs import resolve_run, write_config
from motion_estimator.methods.loftr.supervision import coarse_supervision, fine_supervision

DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"


def run_loftr(model, image0, image1):
    """The kornia LoFTR pipeline, stopping where the losses need to look.

    A transcription of ``LoFTR.forward`` up to the fine matcher, keeping the
    intermediate ``data`` dict instead of discarding it.
    """
    data = {
        "bs": image0.size(0),
        "hw0_i": image0.shape[2:],
        "hw1_i": image1.shape[2:],
    }
    feats_c, feats_f = model.backbone(torch.cat([image0, image1], dim=0))
    (feat_c0, feat_c1) = feats_c.split(data["bs"])
    (feat_f0, feat_f1) = feats_f.split(data["bs"])
    data.update(
        {
            "hw0_c": feat_c0.shape[2:],
            "hw1_c": feat_c1.shape[2:],
            "hw0_f": feat_f0.shape[2:],
            "hw1_f": feat_f1.shape[2:],
        }
    )

    feat_c0 = model.pos_encoding(feat_c0).permute(0, 2, 3, 1)
    n, _, _, c = feat_c0.shape
    feat_c0 = feat_c0.reshape(n, -1, c)
    feat_c1 = model.pos_encoding(feat_c1).permute(0, 2, 3, 1)
    n1, _, _, c1 = feat_c1.shape
    feat_c1 = feat_c1.reshape(n1, -1, c1)

    feat_c0, feat_c1 = model.loftr_coarse(feat_c0, feat_c1, None, None)
    return data, feat_c0, feat_c1, feat_f0, feat_f1


def _pad_with_ground_truth(data, targets, minimum: int) -> None:
    """Top the predicted coarse matches up with ground-truth ones.

    Without this the fine level only ever refines cells the model already
    matches, so it never learns to fix the ones it gets wrong -- and early in
    training it matches almost nothing, leaving the fine head with no gradient
    at all. Upstream LoFTR pads for exactly this reason.

    Padding is appended, not substituted: the predicted matches stay, so the
    fine level still sees its own output distribution alongside the targets.
    """
    predicted = data["b_ids"].numel()
    if predicted >= minimum:
        return
    b_ids, i_ids, j_ids = (
        targets["spv_b_ids"],
        targets["spv_i_ids"],
        targets["spv_j_ids"],
    )
    if b_ids.numel() == 0:
        return
    wanted = minimum - predicted
    pick = torch.randint(b_ids.numel(), (wanted,), device=b_ids.device)
    data["b_ids"] = torch.cat([data["b_ids"], b_ids[pick]])
    data["i_ids"] = torch.cat([data["i_ids"], i_ids[pick]])
    data["j_ids"] = torch.cat([data["j_ids"], j_ids[pick]])
    # mkpts*_c and mconf are recomputed from the padded ids so fine_preprocess
    # and the reported matches stay consistent with each other.
    scale = data["hw0_i"][0] / data["hw0_c"][0]
    width0, width1 = data["hw0_c"][1], data["hw1_c"][1]
    data["mkpts0_c"] = torch.stack(
        [data["i_ids"] % width0, data["i_ids"] // width0], dim=1
    ) * scale
    data["mkpts1_c"] = torch.stack(
        [data["j_ids"] % width1, data["j_ids"] // width1], dim=1
    ) * scale
    if "mconf" in data:
        data["mconf"] = torch.cat(
            [data["mconf"], torch.zeros(wanted, device=data["mconf"].device)]
        )


def step_loss(model, batch, device, args):
    image0 = batch["image0"].to(device, non_blocking=True)
    image1 = batch["image1"].to(device, non_blocking=True)
    points0 = batch["points0"].to(device)
    points1 = batch["points1"].to(device)
    valid = batch["valid"].to(device)

    data, feat_c0, feat_c1, feat_f0, feat_f1 = run_loftr(model, image0, image1)

    targets = coarse_supervision(
        points0, points1, valid, tuple(data["hw0_c"]), tuple(data["hw1_c"])
    )
    data.update(
        {
            "spv_b_ids": targets["spv_b_ids"],
            "spv_i_ids": targets["spv_i_ids"],
            "spv_j_ids": targets["spv_j_ids"],
        }
    )
    # kornia 0.8.3's CoarseMatching.get_coarse_match raises in training mode
    # ("too many values to unpack (expected 2)" -- it unpacks a four-way zip as
    # two), so the matchers run in eval mode and the ground-truth padding it
    # would have done is applied below instead. Patching site-packages was the
    # alternative and is worse: the torch reinstall in this environment already
    # demonstrated how long such an edit survives.
    model.coarse_matching.eval()
    model.fine_matching.eval()
    model.coarse_matching(feat_c0, feat_c1, data)
    if args.gt_padding > 0:
        _pad_with_ground_truth(data, targets, args.gt_padding)

    loss = coarse_focal_loss(
        data["conf_matrix"], targets["conf_matrix_gt"], targets["cell_mask0"]
    )
    parts = {"coarse": float(loss.detach())}

    if args.fine_weight > 0 and data["b_ids"].numel() > 0:
        unfold0, unfold1 = model.fine_preprocess(
            feat_f0, feat_f1, feat_c0, feat_c1, data
        )
        if unfold0.size(0) != 0:
            unfold0, unfold1 = model.loftr_fine(unfold0, unfold1)
        model.fine_matching(unfold0, unfold1, data)
        target_fine = fine_supervision(
            points0,
            points1,
            valid,
            data["b_ids"],
            data["i_ids"],
            tuple(data["hw0_c"]),
        )
        fine = fine_l2_loss(data["expec_f"], target_fine)
        loss = loss + args.fine_weight * fine
        parts["fine"] = float(fine.detach())

    parts["supervised_cells"] = int(targets["cell_mask0"].sum())
    return loss, parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--run-name", default=None,
                        help="label for this run's directory under artifacts/runs/")
    parser.add_argument("--run-dir", default=None,
                        help="write into this existing directory instead of a new one")
    parser.add_argument("--weights", default="outdoor", choices=("outdoor", "indoor"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine-weight", type=float, default=1.0)
    parser.add_argument("--gt-padding", type=int, default=256,
                        help="minimum coarse matches fed to the fine level, "
                             "topped up from ground truth; 0 disables")
    parser.add_argument("--min-spacing", type=int, default=8,
                        help="frame gap floor; below ~8 the motion is smaller "
                             "than one coarse cell and teaches zero displacement")
    parser.add_argument("--max-spacing", type=int, default=20,
                        help="frame gap ceiling; ~1 px of motion per step, so "
                             "this sets the widest baseline trained on")
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        clips = list_cached(args.cache_dir)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1
    train_clips, val_clips = split_by_patient(clips, args.val_fraction, args.seed)
    if not train_clips or not val_clips:
        print(
            f"{len(clips)} cached clips is too few to hold out a patient; "
            "build more of the cache first",
            file=sys.stderr,
        )
        return 1
    print(
        f"{len(clips)} clips: {len(train_clips)} train / {len(val_clips)} val "
        f"({len({c.group_id for c in train_clips})} / "
        f"{len({c.group_id for c in val_clips})} patients)"
    )

    common = dict(
        size=args.size,
        min_spacing=args.min_spacing,
        max_spacing=args.max_spacing,
        n_points=args.n_points,
    )
    train_set = SimPairDataset(
        train_clips, epoch_length=args.steps * args.batch_size, seed=args.seed, **common
    )
    val_set = SimPairDataset(
        val_clips,
        epoch_length=args.val_batches * args.batch_size,
        seed=args.seed + 1,
        augment=False,
        **common,
    )
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, num_workers=args.workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, num_workers=0, drop_last=True
    )

    from kornia.feature import LoFTR

    model = LoFTR(pretrained=args.weights).to(device)
    if not args.unfreeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"training {sum(p.numel() for p in trainable) / 1e6:.1f}M of "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters"
    )

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=args.steps, pct_start=0.1
    )
    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = resolve_run(args.run_dir, args.run_name)
    write_config(out, {"stage": "train", **vars(args), "run_dir": str(out)})
    print(f"run directory: {out}")
    history, best = [], float("inf")
    started = time.time()

    model.train()
    for step, batch in enumerate(loader, 1):
        optimiser.zero_grad(set_to_none=True)
        # fp16, never bf16 -- see the module docstring.
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss, parts = step_loss(model, batch, device, args)
        if not torch.isfinite(loss):
            print(f"  step {step}: non-finite loss, skipped", file=sys.stderr)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        before = scaler.get_scale()
        scaler.step(optimiser)
        scaler.update()
        # Only advance the schedule when the optimiser actually stepped. A
        # GradScaler skips the step on a non-finite gradient and lowers the
        # scale; advancing anyway burns a learning-rate step on nothing and is
        # what torch warns about.
        if scaler.get_scale() >= before:
            schedule.step()

        if step % 25 == 0:
            rate = (time.time() - started) / step
            print(
                f"[{step}/{args.steps}] loss {float(loss):.4f} "
                f"coarse {parts['coarse']:.4f} "
                f"fine {parts.get('fine', float('nan')):.4f} "
                f"cells {parts['supervised_cells']} "
                f"{rate:.2f}s/step eta {(args.steps - step) * rate / 60:.0f}min"
            )
        if step % args.val_every == 0 or step == args.steps:
            value = validate(model, val_loader, device, args)
            history.append({"step": step, "val_loss": value})
            marker = ""
            if value < best:
                best = value
                torch.save(
                    {"model": model.state_dict(), "args": vars(args), "step": step},
                    out / "best.pt",
                )
                marker = "  <- best, saved"
            print(f"  validation {value:.4f}{marker}")
            model.train()
        if step >= args.steps:
            break

    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "last.pt")
    (out / "history.json").write_text(
        json.dumps({"history": history, "args": vars(args)}, indent=2), encoding="utf-8"
    )
    print(f"\nbest validation {best:.4f}; wrote {out}")
    return 0


@torch.no_grad()
def validate(model, loader, device, args) -> float:
    model.eval()
    values = []
    for batch in loader:
        try:
            # Eval mode makes kornia's coarse matcher drop the spv path, but a
            # batch with no coarse match above threshold still has to not crash.
            loss, _ = step_loss(model, batch, device, args)
        except (ValueError, RuntimeError, KeyError):
            continue
        if torch.isfinite(loss):
            values.append(float(loss))
    return float(np.mean(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
