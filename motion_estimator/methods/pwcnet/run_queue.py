"""Run the paper's tests one after another, unattended.

    python -m motion_estimator.methods.pwcnet.run_queue --add MS47_IV0 MS100_IV0
    python -m motion_estimator.methods.pwcnet.run_queue --add MS47_IV85 --folds 0 1 2 3 4 5
    python -m motion_estimator.methods.pwcnet.run_queue --add MS47_IV0 --tag pretrained         --job-args "--net pretrained --steps 30000 --lr-boundaries 15000 20000 25000"
    python -m motion_estimator.methods.pwcnet.run_queue            # work through the queue

``<out-root>/queue.json`` lists the jobs in order. It is re-read before every
job, so jobs can be added or reordered while one is training. A job is
``{"test", "fold", "tag", "args"}``: ``tag`` names a variant (its own run
folder), ``args`` are extra ``train_paper`` arguments for that job only.

* A job whose ``status.json`` says ``done`` is skipped.
* A failed job is retried once, resuming from its ``last.pt``. After a
  second failure it stays failed and the queue moves on. Stopping the runner
  with Ctrl+C is not a failure: the job resumes on the next start.
* Only one runner at a time: ``runner.lock`` holds the PID.
* **The PC does not idle-sleep while the runner is alive** (Windows
  ``SetThreadExecutionState``, system-required). The display can still turn
  off and lock; manual sleep, closing the lid and shutdown still work; nothing
  in the power settings changes, and normal sleep returns when the runner
  exits. Added after the PC slept from 23 to 26 Oct with a run paused at step
  8,400.

``<out-root>/queue.log`` gets one line per event -- ``START``, ``DONE``,
``FAILED``, ``RETRY``, ``FINISHED`` -- which is what a watcher follows. Each
job's own output goes to ``<run>/train.log``.

Everything is written on the local disk (``artifacts/paper-runs``) and copied
to ``--mirror-root`` (the external drive) after every event; the trainer does
the same for its run after every validation. See ``train_paper`` for why.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .experiments import BY_NAME, CV, N_FOLDS
from .train_paper import MIRROR_ROOT, OUT_ROOT, SPLITS, mirror, run_dir

REPO = Path(__file__).resolve().parents[3]
MAX_ATTEMPTS = 2


def _read(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write(path: Path, payload) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def _log(root: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with open(root / "queue.log", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _name(job) -> str:
    test = job["test"] + (f"_{job['tag']}" if job.get("tag") else "")
    return test + ("" if job["fold"] is None else f"/fold{job['fold']}")


def _dir(root: Path, job) -> Path:
    return run_dir(root, job["test"], job["fold"], job.get("tag", ""))


def add_jobs(root: Path, tests, folds, tag: str = "", job_args: str = "") -> None:
    queue = _read(root / "queue.json", {"jobs": [], "train_args": []})
    known = {(j["test"], j["fold"], j.get("tag", "")) for j in queue["jobs"]}
    for test in tests:
        if test not in BY_NAME:
            raise SystemExit(f"unknown test {test!r}")
        wanted = (folds if folds is not None else range(N_FOLDS)) if BY_NAME[test].protocol == CV else [None]
        for fold in wanted:
            if (test, fold, tag) not in known:
                queue["jobs"].append({"test": test, "fold": fold, "tag": tag,
                                      "args": shlex.split(job_args)})
    _write(root / "queue.json", queue)
    print("\n".join(_name(j) for j in queue["jobs"]))


def _lock(root: Path) -> bool:
    lock = root / "runner.lock"
    if lock.is_file():
        pid = int(lock.read_text().strip() or 0)
        if pid and psutil.pid_exists(pid) and pid != os.getpid():
            print(f"another runner is active (PID {pid})")
            return False
    lock.write_text(str(os.getpid()))
    return True


def _sync(root: Path, mirror_root: Path | None, run: Path | None = None) -> None:
    """Queue files, splits and (optionally) one run, copied to the mirror."""
    if mirror_root is None:
        return
    for name in ("queue.json", "queue.log", "queue_state.json"):
        if (root / name).is_file():
            try:
                shutil.copy2(root / name, mirror_root / name)
            except OSError as error:
                print(f"  mirror: could not copy {name} ({error})", flush=True)
    mirror(root / "splits", mirror_root / "splits")
    if run is not None:
        mirror(run, mirror_root / run.relative_to(root))


def _keep_awake(on: bool) -> None:
    """Hold off idle sleep while *on*; a no-op off Windows."""
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    flags = es_continuous | (es_system_required if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def run(root: Path, mirror_root: Path | None) -> int:
    if not _lock(root):
        return 1
    _keep_awake(True)
    env = {**os.environ, "MKL_THREADING_LAYER": "TBB", "PYTHONUNBUFFERED": "1"}
    attempts = _read(root / "queue_state.json", {})
    shutil.copytree(SPLITS, root / "splits", dirs_exist_ok=True)  # the splits the runs used, beside them
    if mirror_root is not None:
        try:
            mirror_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:  # external drive unplugged: train anyway, copies resume once it is back
            print(f"  mirror: {mirror_root} unavailable ({error})", flush=True)
    _log(root, f"RUNNER up, PID {os.getpid()}")
    _sync(root, mirror_root)
    try:
        while True:
            queue = _read(root / "queue.json", {"jobs": [], "train_args": []})
            job = None
            for candidate in queue["jobs"]:
                out = _dir(root, candidate)
                state = _read(out / "status.json", {}).get("state")
                if state != "done" and attempts.get(_name(candidate), 0) < MAX_ATTEMPTS:
                    job = candidate
                    break
            if job is None:
                _log(root, "FINISHED queue is empty")
                _sync(root, mirror_root)
                return 0

            name, out = _name(job), _dir(root, job)
            out.mkdir(parents=True, exist_ok=True)
            attempts[name] = attempts.get(name, 0) + 1
            _write(root / "queue_state.json", attempts)
            _log(root, f"{'START' if attempts[name] == 1 else 'RETRY'} {name}")
            _sync(root, mirror_root)
            command = [sys.executable, "-m", "motion_estimator.methods.pwcnet.train_paper",
                       "--test", job["test"], "--out-root", str(root),
                       "--mirror-root", str(mirror_root or ""), *queue.get("train_args", []),
                       *job.get("args", [])]
            if job.get("tag"):
                command += ["--tag", job["tag"]]
            if job["fold"] is not None:
                command += ["--fold", str(job["fold"])]
            started = time.time()
            with open(out / "train.log", "a", encoding="utf-8") as log:
                log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
                log.flush()
                try:
                    code = subprocess.call(command, cwd=REPO, env=env, stdout=log,
                                           stderr=subprocess.STDOUT)
                except KeyboardInterrupt:
                    # Stopped by hand (Ctrl+C): not a failure, so it must not use up the retry
                    attempts[name] -= 1
                    _write(root / "queue_state.json", attempts)
                    _log(root, f"STOPPED {name} by hand; resumes from last.pt when the runner restarts")
                    raise
            status = _read(out / "status.json", {})
            hours = (time.time() - started) / 3600
            if code == 0 and status.get("state") == "done":
                _log(root, f"DONE {name} best_val_epe={status.get('best_val_epe', float('nan')):.4f} "
                           f"hours={hours:.1f}")
            else:
                _log(root, f"FAILED {name} exit={code} attempt={attempts[name]} "
                           f"error={status.get('error', '')[:150]}")
            _sync(root, mirror_root, out)
    finally:
        _keep_awake(False)
        (root / "runner.lock").unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--mirror-root", type=Path, default=MIRROR_ROOT,
                        help="copy of everything, e.g. the external drive; '' to disable")
    parser.add_argument("--add", nargs="+", default=None, metavar="TEST")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--tag", default="", help="variant name for the added jobs")
    parser.add_argument("--job-args", default="", help="extra train_paper arguments, one string")
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    if args.add:
        add_jobs(args.out_root, args.add, args.folds, args.tag, args.job_args)
        return 0
    mirror_root = args.mirror_root if str(args.mirror_root) not in ("", ".") else None
    return run(args.out_root, mirror_root)


if __name__ == "__main__":
    raise SystemExit(main())
