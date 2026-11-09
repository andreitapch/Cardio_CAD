import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

from cardio_app.jobs import runner

WORKER = "cardio_app.jobs.classify"


def start(label, config):
    folder = runner.workspace()
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    progress, output = folder / "progress.json", folder / "result.pkl"
    command = [sys.executable, "-m", WORKER, "--config", str(folder / "config.json"), "--output", str(output),
               "--progress", str(progress)]
    return runner.start(label, [command], [progress], folder, {"output": output})


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def dump(path, value):
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as handle:
        pickle.dump(value, handle)
    partial.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch
    from cardio_app.services.classification import comparison, single
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["kind"] == "sweep":
        run = comparison.ClassComparisonRun(config["models"], config["settings"])
        runner.report(args.progress, "sweep", 0, run.total)

        def on_model(current):
            dump(args.output, current.snapshot())
            runner.report(args.progress, "sweep", current.done, current.total)

        comparison.run_sweep(run, on_model)
        dump(args.output, run.snapshot())
        runner.report(args.progress, "sweep", run.done, run.total, "done")
        return 0
    runner.report(args.progress, config["class_type"], 0, 1)
    dump(args.output, single.run(config))
    runner.report(args.progress, config["class_type"], 1, 1, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
