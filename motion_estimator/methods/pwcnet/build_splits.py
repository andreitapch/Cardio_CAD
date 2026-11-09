"""Exact training splits for every test in :mod:`.experiments`.

    python -m motion_estimator.methods.pwcnet.build_splits
    python -m motion_estimator.methods.pwcnet.build_splits --tests MS47_IV85 MS0_IV85

Writes, under ``artifacts/splits/paper_replication/``::

    pools.json      every clip of both pools: patient and pair count
    folds.json      the in vivo folds and validation patients, shared by all tests
    <test>.json     the pairs one test trains, validates and tests on, per fold
    summary.csv     pair and patient counts per test and fold, next to the paper's

Rules, the same for every test, so tests differ only in composition:

* **In vivo folds are by patient**, keyed ``<dataset>/<IV##>`` because Retest
  and Test reuse numbers for different people. They are fixed by the seed and
  shared by every CV test, so per-fold results pair up across tests as in the
  paper's t-tests.
* **Validation is by patient on both sides.** In vivo: the first
  ``VAL_FRACTION`` of each fold's training patients, the same in every test.
  Synthetic: a fixed ``VAL_FRACTION`` of patients held out of the pool for
  every test, so the pool fractions apply to what is left.
* **In vivo is taken in whole patients** (the paper's labels are subject
  percentages), in one fixed order, until the test's share of the pool is
  reached. IV85 asks for more than a 6-fold training part holds, so CV tests
  with IV85 take all of it.
* **Synthetic is taken in pairs**, as the paper did, from one fixed permutation
  of the pool, so every smaller subset is contained in every larger one.

A pair is frame ``t`` -> ``t + 1``. Synthetic ``t`` is a 0-based index into the
cached ``frames``. In vivo clips are listed whole; their pairs are
``1 .. n_frames - 1`` in the tree's ``mat/frame<k>.mat`` numbering. Which
``disp/frame<k>.flo`` belongs to which pair (there are ``n_frames + 1`` of
them) is for the in vivo loader to verify, not assumed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np

from . import experiments as ex

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
SIM_CACHE = ARTIFACTS / "simulated-cache"
OUT = ARTIFACTS / "splits" / "paper_replication"
# The paper's own in vivo training tree: frames plus the MITT dense fields.
IV_ROOT = Path(r"F:\PHD\Experiments\Journal_EXP_2022\Datasets"
               r"\2D_GE_intervendor_in_vivo_pxDisp\inVivo\train")
IV_DATASETS = ("Retest_dataset", "Test_dataset")
IV_VIEWS = ("2ch", "4ch")

_FRAME_MAT = re.compile(r"^frame(\d+)\.mat$")


# -- pools ------------------------------------------------------------------

def _npz_frames(path) -> int:
    """Frame count from the ``.npz`` header, without decompressing pixels."""
    with zipfile.ZipFile(path) as archive, archive.open("frames.npy") as handle:
        major, _ = np.lib.format.read_magic(handle)
        read = (np.lib.format.read_array_header_1_0 if major == 1
                else np.lib.format.read_array_header_2_0)
        shape, _, _ = read(handle)
    return int(shape[0])


def synthetic_pool(cache=SIM_CACHE) -> list[dict]:
    """``[{"clip", "patient", "pairs"}]``, ``clip`` relative to the cache."""
    cache = Path(cache)
    pool = [{"clip": path.relative_to(cache).as_posix(),
             "patient": path.parent.name,
             "pairs": _npz_frames(path) - 1}
            for path in sorted(cache.glob("Pat_*/*.npz"))
            if not path.name.endswith(".partial.npz")]
    if not pool:
        raise FileNotFoundError(f"no cached clips under {cache}")
    return pool


def in_vivo_pool(root=IV_ROOT) -> list[dict]:
    """``[{"clip", "patient", "pairs"}]``, ``clip`` = ``<dataset>/<IV##>/<view>``."""
    root = Path(root)
    pool = []
    for dataset in IV_DATASETS:
        for patient in sorted((root / dataset).glob("IV*")):
            for view in IV_VIEWS:
                mats = patient / view / "mat"
                if not mats.is_dir():
                    continue
                n_frames = sum(1 for name in os.listdir(mats) if _FRAME_MAT.match(name))
                if n_frames > 1:
                    pool.append({"clip": f"{dataset}/{patient.name}/{view}",
                                 "patient": f"{dataset}/{patient.name}",
                                 "pairs": n_frames - 1})
    if not pool:
        raise FileNotFoundError(f"no in vivo clips under {root}")
    return pool


# -- splitting --------------------------------------------------------------

def _rng(purpose: str):
    return np.random.default_rng([ex.SEED, *purpose.encode()])


def _shuffled(items, purpose: str) -> list:
    items = sorted(items)
    _rng(purpose).shuffle(items)
    return items


def _patient_pairs(pool) -> dict[str, int]:
    totals: dict[str, int] = {}
    for clip in pool:
        totals[clip["patient"]] = totals.get(clip["patient"], 0) + clip["pairs"]
    return totals


def _n_val(n: int, val_fraction: float) -> int:
    return max(1, round(n * val_fraction)) if val_fraction > 0 else 0


def make_folds(iv_pool, k=ex.N_FOLDS, val_fraction=ex.VAL_FRACTION) -> dict:
    """In vivo patients per fold, plus the split for a model trained on all of them.

    Each ``train`` list is in a fixed random order: a test that wants fewer in
    vivo patients takes a prefix, so smaller in vivo sets sit inside larger ones.
    """
    patients = _shuffled(_patient_pairs(iv_pool), "iv-folds")
    cv = []
    for fold in range(k):
        test = set(patients[fold::k])
        rest = [p for p in patients if p not in test]
        n_val = _n_val(len(rest), val_fraction)
        cv.append({"fold": fold, "test": sorted(test),
                   "val": sorted(rest[:n_val]), "train": rest[n_val:]})
    everyone = _shuffled(patients, "iv-full")
    n_val = _n_val(len(everyone), val_fraction)
    full = {"val": sorted(everyone[:n_val]), "train": everyone[n_val:]}
    return {"k": k, "seed": ex.SEED, "val_fraction": val_fraction,
            "cv": cv, "full": full, "all": sorted(patients)}


def synthetic_order(ms_pool, val_fraction=ex.VAL_FRACTION):
    """Validation patients, and every other pair in one fixed order."""
    patients = _shuffled(_patient_pairs(ms_pool), "ms-val")
    val = set(patients[:_n_val(len(patients), val_fraction)])
    pairs = [(clip["clip"], t) for clip in ms_pool if clip["patient"] not in val
             for t in range(clip["pairs"])]
    order = _rng("ms-pairs").permutation(len(pairs))
    return sorted(val), [pairs[i] for i in order]


def _take_patients(order, sizes, target: int) -> list[str]:
    """The prefix of *order* whose pairs land closest to *target*; all of it if short."""
    taken, total = [], 0
    for patient in order:
        if abs(total + sizes[patient] - target) >= abs(total - target):
            break
        taken.append(patient)
        total += sizes[patient]
    return taken


def build_test(test: ex.Test, ms_pool, ms_val, ms_order, iv_pool, folds) -> dict:
    """Everything one test trains, validates and tests on."""
    n_ms = test.ms_pairs if test.ms_pairs is not None else round(test.ms_fraction * len(ms_order))
    ms_train: dict[str, list[int]] = {}
    for clip, t in sorted(ms_order[:min(n_ms, len(ms_order))]):
        ms_train.setdefault(clip, []).append(t)
    ms_sizes = _patient_pairs(ms_pool)
    ms_val = ms_val if ms_train else []  # no synthetic training, nothing to validate on

    sizes = _patient_pairs(iv_pool)
    clips_of: dict[str, list[str]] = {}
    for clip in iv_pool:
        clips_of.setdefault(clip["patient"], []).append(clip["clip"])
    iv_target = round(test.iv_fraction * sum(sizes.values()))

    if test.protocol == ex.CV:
        parts = folds["cv"]
    elif test.protocol == ex.SUS:
        parts = [{"fold": None, "test": [], **folds["full"]}]
    elif test.protocol == ex.IN_VIVO:
        parts = [{"fold": None, "test": folds["all"], "val": [], "train": []}]
    else:
        raise ValueError(f"{test.name}: unknown protocol {test.protocol!r}")

    rows = []
    for part in parts:
        train = _take_patients(part["train"], sizes, iv_target)
        val = part["val"] if train else []
        rows.append({"fold": part["fold"]})
        for name, patients in (("train", train), ("val", val), ("test", part["test"])):
            rows[-1].update({
                f"iv_{name}": [c for p in sorted(patients) for c in clips_of[p]],
                f"iv_{name}_patients": len(patients),
                f"iv_{name}_pairs": sum(sizes[p] for p in patients),
            })
    return {
        "test": {**vars(test), "ms_fraction": test.ms_fraction, "iv_fraction": test.iv_fraction},
        "seed": ex.SEED,
        "external_test": "SUS" if test.protocol == ex.SUS else None,
        "ms": {"train": ms_train,
               "train_pairs": sum(map(len, ms_train.values())),
               "train_patients": len({clip.split("/")[0] for clip in ms_train}),
               "val_patients": ms_val,
               "val_pairs": sum(ms_sizes[p] for p in ms_val)},
        "folds": rows,
    }


# -- output -----------------------------------------------------------------

SUMMARY_FIELDS = ("test", "protocol", "fold", "paper_ms", "paper_iv",
                  "ms_train_pairs", "ms_train_patients", "ms_val_pairs",
                  "iv_train_pairs", "iv_train_patients", "iv_val_pairs",
                  "iv_test_pairs", "iv_test_patients", "train_pairs", "ms_share")


def summary_rows(split: dict) -> list[dict]:
    test, ms = split["test"], split["ms"]
    rows = []
    for fold in split["folds"]:
        train = ms["train_pairs"] + fold["iv_train_pairs"]
        rows.append({
            "test": test["name"], "protocol": test["protocol"],
            "fold": "" if fold["fold"] is None else fold["fold"],
            "paper_ms": test["ms"], "paper_iv": test["iv"],
            "ms_train_pairs": ms["train_pairs"], "ms_train_patients": ms["train_patients"],
            "ms_val_pairs": ms["val_pairs"],
            "iv_train_pairs": fold["iv_train_pairs"], "iv_train_patients": fold["iv_train_patients"],
            "iv_val_pairs": fold["iv_val_pairs"],
            "iv_test_pairs": fold["iv_test_pairs"], "iv_test_patients": fold["iv_test_patients"],
            "train_pairs": train,
            "ms_share": round(ms["train_pairs"] / train, 3) if train else 0.0,
        })
    return rows


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests", nargs="+", default=None, choices=list(ex.BY_NAME),
                        help="default: every test in experiments.TESTS")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sim-cache", type=Path, default=SIM_CACHE)
    parser.add_argument("--iv-root", type=Path, default=IV_ROOT)
    args = parser.parse_args(argv)

    ms_pool = synthetic_pool(args.sim_cache)
    iv_pool = in_vivo_pool(args.iv_root)
    iv_pairs = sum(clip["pairs"] for clip in iv_pool)
    print(f"synthetic: {len(ms_pool)} clips, {len(_patient_pairs(ms_pool))} patients, "
          f"{sum(c['pairs'] for c in ms_pool)} pairs")
    print(f"in vivo:   {len(iv_pool)} clips, {len(_patient_pairs(iv_pool))} patients, "
          f"{iv_pairs} pairs (paper: {ex.PAPER_IV_POOL})")
    if iv_pairs != ex.PAPER_IV_POOL:
        print("  WARNING: the in vivo pool differs from the paper's; IV counts will not match")

    folds = make_folds(iv_pool)
    ms_val, ms_order = synthetic_order(ms_pool)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(args.out / "pools.json", {
        "synthetic": {"root": str(args.sim_cache), "clips": ms_pool},
        "in_vivo": {"root": str(args.iv_root), "clips": iv_pool},
    })
    _write_json(args.out / "folds.json", folds)

    for name in args.tests or list(ex.BY_NAME):
        split = build_test(ex.BY_NAME[name], ms_pool, ms_val, ms_order, iv_pool, folds)
        _write_json(args.out / f"{name}.json", split)

    # Every test on disk, not only the ones rebuilt now, so --tests never shrinks the summary
    rows = []
    for name in ex.BY_NAME:
        path = args.out / f"{name}.json"
        if path.is_file():
            rows += summary_rows(json.loads(path.read_text(encoding="utf-8")))

    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    shown = ("test", "fold", "paper_ms", "ms_train_pairs", "paper_iv",
             "iv_train_pairs", "iv_train_patients", "iv_test_patients", "ms_share")
    print("\n" + "  ".join(f"{c:>17}" if i == 0 else f"{c:>10}" for i, c in enumerate(shown)))
    for row in rows:
        print("  ".join(f"{row[c]!s:>17}" if i == 0 else f"{row[c]!s:>10}"
                        for i, c in enumerate(shown)))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
