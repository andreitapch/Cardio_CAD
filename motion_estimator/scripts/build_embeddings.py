"""Run EchoCLIP over the cohort once and cache the embeddings.

    python -m motion_estimator.scripts.build_embeddings
    python -m motion_estimator.scripts.build_embeddings --datasets Retest_dataset
    python -m motion_estimator.scripts.build_embeddings --sampling upstream \
        --cache-dir motion_estimator/artifacts/embeddings-upstream

Resumable: clips already cached are reused, so an interrupted pass can simply
be re-run. Use ``--overwrite`` to force a rebuild, which is also what changing
encoder settings in place requires.

The pass is I/O-bound, not GPU-bound -- inference is ~60 ms per clip against
~15 MB of ``data.mat`` to read for each one.
"""
from __future__ import annotations

import argparse
import sys
import time

from motion_estimator.loaders.dataset_index import DATASET_FOLDERS, index_dataset
from motion_estimator.echoclip.config import (
    DEFAULT_CACHE_DIR,
    SAMPLING_STRATEGIES,
    EncoderConfig,
)
from motion_estimator.echoclip.embed_cache import build_cache
from motion_estimator.echoclip.encoder import EchoClipEncoder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="dataset root")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FOLDERS))
    parser.add_argument("--vendor", default="GE", help="ground-truth vendor")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--sampling", choices=SAMPLING_STRATEGIES, default="uniform")
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "fp32", "bf16"),
        help="fp16 on Turing; bf16 has no kernel there and is rejected",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="encode only the first N clips"
    )
    args = parser.parse_args(argv)

    try:
        refs = index_dataset(args.root, tuple(args.datasets), args.vendor)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    complete = [ref for ref in refs if ref.is_complete]
    if args.limit:
        refs = refs[: args.limit]
        complete = [ref for ref in refs if ref.is_complete]

    print(f"{len(refs)} clips indexed, {len(complete)} complete")
    print(f"cache: {args.cache_dir}")

    encoder = EchoClipEncoder(
        EncoderConfig(
            precision=args.precision,
            device=args.device,
            n_frames=args.n_frames,
            sampling=args.sampling,
        )
    )

    print("loading EchoCLIP ...", flush=True)
    started = time.time()
    try:
        encoder.load()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"  ready on {encoder.device} in {time.time() - started:.1f}s")

    state = {"done": 0, "started": time.time()}

    def progress(key, action):
        state["done"] += 1
        if action == "written" or state["done"] % 25 == 0:
            elapsed = time.time() - state["started"]
            rate = state["done"] / elapsed if elapsed else 0
            remaining = (len(complete) - state["done"]) / rate if rate else 0
            print(
                f"  [{state['done']:>3}/{len(complete)}] {action:<7} {key} "
                f"({rate:.1f} clips/s, ~{remaining / 60:.1f} min left)",
                flush=True,
            )

    report = build_cache(
        refs, encoder, args.cache_dir, overwrite=args.overwrite, progress=progress
    )

    print()
    print(report.summary())
    print(f"total {time.time() - started:.1f}s")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
