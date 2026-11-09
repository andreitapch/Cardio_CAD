"""Train a head on the cached embeddings and compare it against the baselines.

    python -m motion_estimator.scripts.train_head
    python -m motion_estimator.scripts.train_head --head mean --epochs 500

Development runs use Retest_dataset only; --final is required to touch the
held-out Test_dataset and exists to make that a deliberate act.
"""
from __future__ import annotations

import argparse
import sys
import time

from motion_estimator.loaders.dataset_index import index_dataset
from motion_estimator.echoclip.config import DEFAULT_CACHE_DIR
from motion_estimator.echoclip.embed_cache import GROUPINGS, load_embeddings
from motion_estimator.echoclip.metrics import TARGET_LABELS
from motion_estimator.echoclip.probe import run_cv
from motion_estimator.echoclip.train import TrainConfig, train_cv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--phase", default="end_systolic")
    parser.add_argument("--grouping", choices=GROUPINGS, default="dataset_patient")
    parser.add_argument("--head", choices=("attn", "mean"), default="attn")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)

    try:
        refs = index_dataset(args.root)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    data = load_embeddings(refs, args.cache_dir, args.phase, args.grouping)
    if not args.final:
        data = data.subset(data.datasets == "Retest_dataset")
    else:
        print("*** using the held-out Test_dataset ***")

    print(f"clips {len(data)}  groups {data.n_groups}  patients {data.n_patients}")
    print(
        "support  "
        + "  ".join(f"{a}={b}" for a, b in zip(TARGET_LABELS, data.support()))
    )

    config = TrainConfig(
        head=args.head,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden=args.hidden,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    print(f"\n{config.describe()}")

    print("\n" + "=" * 78)
    results = [
        run_cv(data, "mean", n_splits=args.n_splits),
        run_cv(data, "ridge", alpha=args.ridge_alpha, n_splits=args.n_splits),
    ]
    started = time.time()
    results.append(train_cv(data, config, n_splits=args.n_splits))
    elapsed = time.time() - started

    for result in results:
        print(result.report())
        print()
    print(f"head trained in {elapsed:.1f}s")

    print("=" * 78)
    print(f"{'target':6} " + "".join(f"{r.name:>22}" for r in results))
    for index, label in enumerate(TARGET_LABELS):
        row = f"{label:6} " + "".join(
            f"{r.metrics[index].mae:>22.2f}" for r in results
        )
        print(row)
    print("\n(MAE, out-of-fold, grouped CV. Lower is better.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
