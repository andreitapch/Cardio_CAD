"""One directory per experiment, so runs cannot overwrite each other.

The first fine-tune wrote straight into ``artifacts/loftr-finetuned/``, which
means a second run silently replaces the first -- including the checkpoint you
would want to compare against. Worse, the evaluation JSONs landed in the repo
root with hand-chosen names, so which checkpoint produced which numbers was
recorded nowhere but the filename.

A run directory fixes both: the config, the checkpoints, the loss history and
the evaluation results for one experiment sit together, and the directory name
carries the date so the ordering is obvious without opening anything.

    artifacts/runs/2026-09-15_1432_spacing-floor/
        config.json     every argument the run was started with
        history.json    the validation curve
        best.pt         lowest validation loss
        last.pt         final step
        eval_*.json     evaluation results, written next to the weights

``--resume`` and ``--run-dir`` both take an existing directory, so a re-run
adds to it rather than starting a new one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["DEFAULT_RUNS_ROOT", "create_run", "resolve_run", "write_config", "latest_run"]

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "runs"


def create_run(name: str | None = None, root: str | Path | None = None) -> Path:
    """A fresh, empty run directory named ``<date>_<time>_<name>``.

    The timestamp goes first so a directory listing sorts chronologically. If
    the name is already taken within the same minute -- two runs launched by a
    script, say -- a counter is appended rather than reusing the directory.
    """
    root = Path(root) if root is not None else DEFAULT_RUNS_ROOT
    stamp = time.strftime("%Y-%m-%d_%H%M")
    label = f"{stamp}_{_slug(name)}" if name else stamp

    candidate = root / label
    counter = 2
    while candidate.exists():
        candidate = root / f"{label}-{counter}"
        counter += 1
    candidate.mkdir(parents=True)
    return candidate


def resolve_run(run_dir=None, name=None, root=None) -> Path:
    """An existing directory if given, otherwise a new one."""
    if run_dir is not None:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return create_run(name, root)


def latest_run(root: str | Path | None = None) -> Path | None:
    """The most recent run directory, or None if there are none.

    Used so evaluation can default to "the run I just trained" instead of
    making you paste a path.
    """
    root = Path(root) if root is not None else DEFAULT_RUNS_ROOT
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Name-sorted, because the timestamp prefix makes that chronological and
    # mtime changes whenever anything is written into a directory.
    return sorted(candidates, key=lambda p: p.name)[-1]


def write_config(run_dir: Path, config: dict) -> Path:
    """Record what the run was started with, merging into any existing config."""
    path = Path(run_dir) / "config.json"
    existing = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    merged = {**existing, **config, "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    path.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    return path


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(name).strip()]
    return "".join(keep).strip("-").lower() or "run"
