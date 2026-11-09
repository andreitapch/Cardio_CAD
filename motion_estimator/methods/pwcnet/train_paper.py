"""Train PWC-Net for one test (and fold) of the paper's replication.

    python -m motion_estimator.methods.pwcnet.train_paper --test MS47_IV0
    python -m motion_estimator.methods.pwcnet.train_paper --test MS47_IV85 --fold 3
    python -m motion_estimator.methods.pwcnet.train_paper --test MS47_IV0 --tag pretrained --net pretrained

Everything for a run lands in ``<out-root>/<test>[_<tag>]/<fold0..5 | single>/``
on the local disk (``--tag`` keeps a variant, e.g. the pretrained runs, apart), and is copied to the same place under ``--mirror-root`` (the
external drive) after every validation and at the end:

    config.json    every argument, the split's counts, the code that ran
    history.json   validation after every --val-every steps
    status.json    running / done / failed, step, best -- what the queue reads
    best.pt        lowest validation EPE (model only)
    step<N>.pt     the model at each --snapshots step, for equal-step comparisons
    eval_invivo*.json  the paper metric on the test patients, for best.pt and each snapshot,
                   computed when the run finishes
    last.pt        model + optimiser + scaler + step, for resuming

A run resumes from ``last.pt`` when it exists, so a crash or reboot costs at
most --val-every steps. **Extending a finished run** is the same command with a
larger ``--steps``: the learning rate depends on the step alone (constant 1e-4,
halved at 100k, 150k, 200k, 250k, 300k -- the old TF1 synthetic schedule), so an
extension continues the schedule exactly where an uninterrupted run would be.

**Why local first.** The first trial wrote straight to F: and died at step 500
with ``PermissionError(13)`` on a plain log write -- the external drive refused
a write while the file was being read. A copy that fails is retried and then
skipped with a warning; it never stops training. The next copy catches up.

Training follows the paper's setup: synthetic and in vivo pairs mixed in one
pool in the proportion the split gives, sampled uniformly; consecutive pairs
only; dense multiscale end-point loss with the old echo runs' level weights.
``best.pt`` is chosen on the mean of the per-source validation EPEs.

**The paper's metric during validation.** Every ``--paper-val-every`` steps
(5,000) the fold's in vivo validation clips (2ch and 4ch) are tracked end to
end and scored with the paper's contour EPE in mm (:mod:`.paper_metrics`,
corrected spacing and paper-exact). It is logged in ``history.json`` as
``val_paper_epe_mm`` / ``val_paper_epe_mm_exact``; it does not pick
``best.pt`` (the dense EPE does, as in the TF1 training).

**From scratch, as the paper** (``--net scratch``, :class:`.net.PWCNet`, the
rewrite of the TF1 architecture). Chosen by Andrea over the alternative,
``--net pretrained`` (:mod:`.net_pretrained`, FlyingChairs weights), which
fine-tunes far faster: from scratch the first 5,000 steps sat at the zero-flow
error, pretrained reached 0.26 px within 1,000 steps. The old TF1 runs trained
up to 600k steps with their best checkpoints at 50-90k, hence the long budget.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .losses import LEVEL_WEIGHTS
from .net import PWCNet
from .net_pretrained import IRR_CHECKPOINT, PretrainedPWCNet
from .paper_cache import PAPER_CACHE
from .paper_data import IV, MS, PaperPairDataset, _fold, load_split, train_items, val_items

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SPLITS = ARTIFACTS / "splits" / "paper_replication"
OUT_ROOT = ARTIFACTS / "paper-runs"
MIRROR_ROOT = Path(r"F:\PHD\Experiments\PWC_Replication_2026")


def run_dir(out_root, test: str, fold: int | None, tag: str = "") -> Path:
    name = f"{test}_{tag}" if tag else test
    return Path(out_root) / name / ("single" if fold is None else f"fold{fold}")


def mirror(source: Path, target: Path, attempts: int = 3) -> bool:
    """Copy every new or changed file under *source* to *target*; never raises."""
    ok = True
    for path in [p for p in Path(source).rglob("*") if p.is_file() and not p.name.endswith(".partial")
                 and ".partial." not in p.name]:
        destination = Path(target) / path.relative_to(source)
        for attempt in range(attempts):
            try:
                stat = path.stat()
                if destination.is_file():
                    have = destination.stat()
                    if have.st_size == stat.st_size and have.st_mtime >= stat.st_mtime:
                        break
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination.with_name(destination.name + ".copying"))
                destination.with_name(destination.name + ".copying").replace(destination)
                break
            except OSError as error:
                if attempt == attempts - 1:
                    print(f"  mirror: could not copy {path.name} ({error})", file=sys.stderr, flush=True)
                    ok = False
                else:
                    time.sleep(2)
    return ok


def dense_loss(output, flow, mask, weights=LEVEL_WEIGHTS):
    """Multiscale masked end-point error.

    At each pyramid level the target is the mask-weighted mean of the
    full-resolution flow in the cell, and the cell counts in proportion to its
    myocardium coverage -- a thin wall covers few coarse cells fully, and a
    hard threshold would drop the coarse levels altogether. Returns the loss
    and the full-resolution EPE per sample ``(B,)``.
    """
    loss = 0.0
    for level, predicted in output["pyramid"]:
        scale = 2 ** level
        cover = F.avg_pool2d(mask, scale)
        target = F.avg_pool2d(flow * mask, scale) / cover.clamp_min(1e-6)
        error = torch.linalg.norm(predicted.float() - target, dim=1, keepdim=True)
        loss = loss + weights.get(level, 0.0) * (error * cover).sum() / cover.sum().clamp_min(1e-6)
    error = torch.linalg.norm(output["flow"].float() - flow, dim=1, keepdim=True)
    per_sample = (error * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return loss, per_sample


@torch.no_grad()
def validate(model, loader, device, amp) -> dict:
    model.eval()
    epes = {MS: [], IV: []}
    for batch in loader:
        image0, image1 = batch["image0"].to(device), batch["image1"].to(device)
        flow, mask = batch["flow"].to(device), batch["mask"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(image0, image1)
        _, per_sample = dense_loss(output, flow, mask)
        for source, epe in zip(batch["source"].tolist(), per_sample.tolist()):
            epes[source].append(epe)
    model.train()
    result = {}
    if epes[MS]:
        result["val_epe_ms"] = float(np.mean(epes[MS]))
    if epes[IV]:
        result["val_epe_iv"] = float(np.mean(epes[IV]))
    result["val_epe"] = float(np.mean(list(result.values())))
    return result


def _write(path: Path, payload) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _status(out: Path, state: str, **extra) -> None:
    _write(out / "status.json", {"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 **extra})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--tag", default="", help="variant name appended to the run folder")
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mirror-root", type=Path, default=MIRROR_ROOT,
                        help="copy of every run, e.g. the external drive; '' to disable")
    parser.add_argument("--cache", type=Path, default=PAPER_CACHE)
    parser.add_argument("--net", choices=("pretrained", "scratch"), default="scratch")
    parser.add_argument("--pretrained", type=Path, default=IRR_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=150000, help="total; raise it to extend a run")
    parser.add_argument("--snapshots", type=int, nargs="*", default=[30000, 60000, 100000, 150000],
                        help="steps at which to keep a copy of the model (step<N>.pt)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-boundaries", type=int, nargs="+",
                        default=[100000, 150000, 200000, 250000, 300000], help="lr halves at each")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=384, help="crop, multiple of 64")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-pairs", type=int, default=512, help="per source")
    parser.add_argument("--paper-val-every", type=int, default=5000,
                        help="steps between paper-EPE validations on whole clips; 0 disables")
    parser.add_argument("--paper-val-views", nargs="+", default=["2ch", "4ch"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args(argv)

    split, pools = load_split(args.splits, args.test)
    out = run_dir(args.out_root, args.test, args.fold, args.tag)
    out.mkdir(parents=True, exist_ok=True)
    mirrored = str(args.mirror_root) not in ("", ".")
    copy = lambda: mirror(out, run_dir(args.mirror_root, args.test, args.fold, args.tag)) if mirrored else True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    train = train_items(split, pools, args.fold)
    paper_val_keys = [k for k in _fold(split, args.fold)["iv_val"]
                      if k.rsplit("/", 1)[1] in args.paper_val_views]
    val = val_items(split, pools, args.fold, args.val_pairs, seed=args.seed)
    counts = {"train_ms": sum(s == MS for s, _, _ in train), "train_iv": sum(s == IV for s, _, _ in train),
              "val_ms": sum(s == MS for s, _, _ in val), "val_iv": sum(s == IV for s, _, _ in val)}
    if not train:
        raise SystemExit(f"{args.test}: no training pairs")

    model = (PretrainedPWCNet(args.pretrained) if args.net == "pretrained" else PWCNet()).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=4e-4)
    lr_at = lambda at: args.lr * 0.5 ** sum(at >= b for b in args.lr_boundaries)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    step, best, history = 0, float("inf"), []
    if (out / "last.pt").is_file():
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if state["step"] >= args.steps:
            print(f"{out} already at step {state['step']}")
            _status(out, "done", step=state["step"], best_val_epe=state["best"])
            return 0
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scaler.load_state_dict(state["scaler"])
        step, best, history = state["step"], state["best"], state["history"]
        print(f"resumed from step {step} (best val EPE {best:.4f})"
              + (" -- extending a finished run" if state.get("finished") else ""))

    _write(out / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "split": split["test"], "counts": counts, "external_test": split["external_test"],
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "level_weights": LEVEL_WEIGHTS, "loss": "dense multiscale EPE, coverage-weighted",
        "optimiser": "AdamW wd 4e-4, lr halved at each --lr-boundaries step, grad clip 5",
    })
    print(f"{args.test} fold {args.fold}: {counts} -> {out}", flush=True)

    generator = torch.Generator().manual_seed(args.seed * 1_000_003 + step)
    sampler = torch.utils.data.RandomSampler(range(len(train)), replacement=True,
                                             num_samples=(args.steps - step) * args.batch_size,
                                             generator=generator)
    loader = torch.utils.data.DataLoader(
        PaperPairDataset(train, args.size, train=True, cache=args.cache),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        drop_last=True, persistent_workers=args.workers > 0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        PaperPairDataset(val, args.size, train=False, cache=args.cache),
        batch_size=args.batch_size, num_workers=args.workers)

    def checkpoint(finished=False):
        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step, "best": best, "history": history, "finished": finished,
                    "args": vars(args)}, out / "last.partial.pt")
        (out / "last.partial.pt").replace(out / "last.pt")

    _status(out, "running", step=step, best_val_epe=best)
    model.train()
    started, first_step, running = time.time(), step, []
    try:
        for batch in loader:
            step += 1
            image0 = batch["image0"].to(device, non_blocking=True)
            image1 = batch["image1"].to(device, non_blocking=True)
            flow = batch["flow"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            for group in optimiser.param_groups:
                group["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                output = model(image0, image1)
            loss, per_sample = dense_loss(output, flow, mask)
            if torch.isfinite(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimiser)
                scaler.update()
                running.append((float(loss.detach()), float(per_sample.detach().mean())))
            else:
                print(f"  step {step}: non-finite loss, skipped", file=sys.stderr, flush=True)

            if step % 100 == 0 and running:
                rate = (time.time() - started) / (step - first_step)
                loss_avg, epe_avg = np.mean(running, axis=0)
                running = []
                print(f"[{step}/{args.steps}] loss {loss_avg:.4f} epe {epe_avg:.3f}px "
                      f"lr {lr_at(step):.1e} {rate:.2f}s/step "
                      f"eta {(args.steps - step) * rate / 3600:.1f}h", flush=True)
            if step % args.val_every == 0 or step == args.steps:
                result = validate(model, val_loader, device, amp)
                if paper_val_keys and args.paper_val_every and step % args.paper_val_every == 0:
                    # Imported here: paper_evaluate imports this module
                    from .paper_evaluate import evaluate_clips, summaries
                    model.eval()
                    paper = summaries(evaluate_clips(model, paper_val_keys, device, amp))
                    model.train()
                    result["val_paper_epe_mm"] = paper["all_corrected"]["mean"]
                    result["val_paper_epe_mm_exact"] = paper["all_paper_exact"]["mean"]
                    for view in args.paper_val_views:
                        if f"{view}_corrected" in paper:
                            result[f"val_paper_epe_mm_{view}"] = paper[f"{view}_corrected"]["mean"]
                history.append({"step": step, "lr": lr_at(step), **result})
                marker = ""
                if result["val_epe"] < best:  # the dense EPE picks best.pt, not the paper metric
                    best = result["val_epe"]
                    torch.save({"model": model.state_dict(), "step": step, **result,
                                "test": args.test, "fold": args.fold}, out / "best.pt")
                    marker = "  <- best"
                print("  val " + " ".join(f"{k} {v:.4f}" for k, v in result.items()) + marker,
                      flush=True)
                if step in args.snapshots:
                    torch.save({"model": model.state_dict(), "step": step, **result,
                                "test": args.test, "fold": args.fold}, out / f"step{step}.pt")
                _write(out / "history.json", {"history": history})
                checkpoint()
                _status(out, "running", step=step, best_val_epe=best,
                        eta_h=round((args.steps - step) * (time.time() - started)
                                    / max(step - first_step, 1) / 3600, 2))
                copy()
            if step >= args.steps:
                break
    except BaseException as error:
        _status(out, "failed", step=step, best_val_epe=best, error=repr(error))
        copy()
        raise

    checkpoint(finished=True)
    hours = (time.time() - started) / 3600
    # The paper's metric on the held-out test patients for best.pt and every snapshot, so each
    # finished run has its test number and the equal-step comparison. A failure here must not
    # undo a finished training, so it only prints.
    from .paper_evaluate import main as evaluate_test
    for name in ["best.pt"] + sorted(p.name for p in out.glob("step*.pt")):
        try:
            evaluate_test([str(out), "--checkpoint", name, "--batch", "4"])
        except Exception as error:
            print(f"  test evaluation of {name} failed ({error!r}); run paper_evaluate by hand",
                  file=sys.stderr, flush=True)
    _status(out, "done", step=step, best_val_epe=best, hours=round(hours, 2))
    print(f"done: best val EPE {best:.4f} px, {hours:.1f} h, {out}", flush=True)
    copy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
