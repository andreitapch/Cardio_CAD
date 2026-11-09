import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

from cardio_app.jobs import runner
from cardio_app.services import reduction

WORKER = "cardio_app.jobs.reduce"


def start(label, items, methods, k, epochs):
    folder = runner.workspace()
    reduction.save_blocks(folder / "input.npz", items)
    progress, output = folder / "progress.json", folder / "result.npz"
    command = [sys.executable, "-m", WORKER, "--input", str(folder / "input.npz"), "--output", str(output),
               "--methods", *methods, "--k", str(k), "--epochs", str(epochs), "--progress", str(progress)]
    return runner.start(label, [command], [progress], folder, {"output": output})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    blocks = reduction.load_blocks(args.input)
    weights = {"PCA": 1, "AutoencoderDm": args.epochs}
    total = sum(weights[m] * len(blocks) for m in args.methods)
    done = 0
    results = []
    for method in args.methods:
        result = reduction.Result(method)
        for name, x in blocks.items():
            if method == "PCA":
                codes, recon, metrics, params = reduction.reduce_pca(x, args.k)
                result.models[name] = params
                done += 1
                runner.report(args.progress, method, done, total)
            else:
                base = done

                def progress(epoch):
                    runner.report(args.progress, method, base + epoch, total)

                codes, recon, metrics, history = reduction.reduce_autoencoder(x, args.k, args.epochs, progress)
                result.history[name] = history
                done = base + args.epochs
            result.codes[name], result.recon[name], result.metrics[name], result.k[name] = codes, recon, metrics, metrics["k"]
        results.append(result)
    reduction.save_result(args.output, results)
    runner.report(args.progress, "+".join(args.methods), total, total, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
