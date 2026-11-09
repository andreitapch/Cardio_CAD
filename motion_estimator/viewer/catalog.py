"""What the viewer can show: the in vivo clips, and the trained models.

No torch here -- the app imports this, and torch beside the app's scientific
stack aborts the process (two OpenMP runtimes).

**Clips** are the 236 in vivo recordings (Retest + Test, 2ch + 4ch) in
``artifacts/invivo-cache``: one cardiac cycle from ``data.mat``, MITT's
contour per frame, the pixel spacing and GE's GLS.

**Models** are the trained runs under ``artifacts/paper-runs``, one per test
and variant (``MS47_IV0``, ``MS47_IV0_pretrained`` ...). A cross-validated
test is one model across its folds: each patient is tracked by the fold whose
held-out patients include it, so every prediction is on unseen data. A
patient whose fold has not been trained yet has no prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
INVIVO_CACHE = ARTIFACTS / "invivo-cache"
RUNS = ARTIFACTS / "paper-runs"
SPLITS = ARTIFACTS / "splits" / "paper_replication"
VIEWER_CACHE = ARTIFACTS / "viewer-cache"

MITT = "MITT"


@dataclass
class Model:
    id: str                                         # run folder name, e.g. MS47_IV85_pretrained
    test: str
    variant: str                                    # "scratch" or the tag
    runs: dict[str, Path] = field(default_factory=dict)   # "single" or "fold<k>" -> run dir with best.pt
    fold_of: dict[str, str] = field(default_factory=dict)  # patient group -> "fold<k>" (CV only)
    trained_on: set[str] = field(default_factory=set)      # patient groups a single run trained on

    @property
    def label(self) -> str:
        if "single" in self.runs:
            return f"{self.test} - {self.variant}"
        return f"{self.test} - {self.variant} (CV, folds {', '.join(sorted(k[4:] for k in self.runs))})"

    def run_for(self, clip_key: str) -> Path | None:
        """The run that tracks this clip on unseen data, or None if not trained yet."""
        if "single" in self.runs:
            return self.runs["single"]
        group = clip_key.rsplit("/", 1)[0]
        return self.runs.get(self.fold_of.get(group, ""))

    def seen(self, clip_key: str) -> bool:
        """True when the model trained on this patient (a single run that used in vivo data)."""
        return clip_key.rsplit("/", 1)[0] in self.trained_on


def clip_keys() -> list[str]:
    """``<dataset>/<IV##>/<view>`` for every cached in vivo clip."""
    keys = []
    for path in sorted(INVIVO_CACHE.glob("*_dataset/IV*_*.npz")):
        patient, view = path.stem.rsplit("_", 1)
        keys.append(f"{path.parent.name}/{patient}/{view}")
    return keys


def clip_file(key: str) -> Path:
    dataset, patient, view = key.split("/")
    return INVIVO_CACHE / dataset / f"{patient}_{view}.npz"


def models() -> dict[str, Model]:
    found: dict[str, Model] = {}
    for best in sorted(RUNS.glob("*/*/best.pt")):
        run, name = best.parent, best.parent.parent.name
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        test, tag = config["test"], config.get("tag", "")
        model = found.setdefault(name, Model(id=name, test=test, variant=tag or "scratch"))
        model.runs[run.name] = run
    for model in found.values():
        split = json.loads((SPLITS / f"{model.test}.json").read_text(encoding="utf-8"))
        if "single" in model.runs:
            model.trained_on = {k.rsplit("/", 1)[0] for k in split["folds"][0]["iv_train"]}
            continue
        for fold in split["folds"]:
            for key in fold["iv_test"]:
                model.fold_of[key.rsplit("/", 1)[0]] = f"fold{fold['fold']}"
    return found


def result_file(model_id: str, clip_key: str) -> Path:
    dataset, patient, view = clip_key.split("/")
    return VIEWER_CACHE / model_id / dataset / f"{patient}_{view}.npz"


def flow_file(model_id: str, clip_key: str) -> Path:
    return result_file(model_id, clip_key).with_suffix(".flow.npy")
