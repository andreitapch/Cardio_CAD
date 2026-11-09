"""Status of every queued training: progress, time left and an estimated finish.

    python -m motion_estimator.methods.pwcnet.queue_status

Reads ``queue.json``, ``queue.log`` and each run's ``status.json`` /
``history.json`` under ``artifacts/paper-runs``. Time left for a queued run is
its step budget at the speed measured so far for its kind (from scratch or
pretrained, with or without in vivo data), so estimates assume no pauses and
no other GPU work. No torch, so it runs beside a training.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "artifacts" / "paper-runs"
# seconds per step measured on the RTX 2060, plus the paper-metric validation every 5k steps (mixed runs)
SPEED = {("scratch", False): 0.50, ("scratch", True): 0.59, ("pretrained", False): 0.48, ("pretrained", True): 0.55}
PAPER_VAL_S = 90


def _stamp(line: str) -> dt.datetime:
    return dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")


def rows(root: Path = RUNS) -> list[dict]:
    queue = json.loads((root / "queue.json").read_text(encoding="utf-8"))["jobs"]
    log = (root / "queue.log").read_text(encoding="utf-8").splitlines()
    now = dt.datetime.now()
    out = []
    for job in queue:
        tag = job.get("tag", "")
        name = job["test"] + (f"_{tag}" if tag else "")
        sub = "single" if job["fold"] is None else f"fold{job['fold']}"
        label = name if job["fold"] is None else f"{name}/{sub}"
        args = job.get("args", [])
        total = int(args[args.index("--steps") + 1]) if "--steps" in args else 100000
        kind = "pretrained" if "pretrained" in args else "scratch"
        mixed = "_IV0" not in job["test"]
        status_file = root / name / sub / "status.json"
        status = json.loads(status_file.read_text(encoding="utf-8")) if status_file.is_file() else {}
        step = int(status.get("step", 0))
        event = re.compile(rf"(START|RETRY|DONE) {re.escape(label)}(\s|$)")
        events = [(m.group(1), _stamp(line)) for line in log if (m := event.search(line))]
        started = next((t for e, t in events if e in ("START", "RETRY")), None)
        finished = next((t for e, t in reversed(events) if e == "DONE"), None)
        per_step = SPEED[(kind, mixed)]
        overhead = (total // 5000) * PAPER_VAL_S if mixed else 0
        if status.get("state") == "done":
            left, finish, state = 0.0, finished, "done"
        else:
            left = ((total - step) * per_step + overhead * (total - step) / total) / 3600
            state = status.get("state") if status.get("state") in ("running", "failed") else "queued"
            finish = None  # filled below: the running job first, then the rest in queue order
        history_file = root / name / sub / "history.json"
        history = json.loads(history_file.read_text(encoding="utf-8"))["history"] if history_file.is_file() else []
        paper = [h["val_paper_epe_mm"] for h in history if "val_paper_epe_mm" in h]
        test_file = root / name / sub / "eval_invivo.json"
        test = json.loads(test_file.read_text(encoding="utf-8"))["summary"]["all_corrected"] if test_file.is_file() else None
        out.append({"run": label, "state": state, "step": step, "total": total,
                    "percent": 100 * step / total, "started": started, "left_h": left, "finish": finish,
                    "best_val_px": status.get("best_val_epe"), "paper_val_mm": min(paper) if paper else None,
                    "paper_test_mm": test["mean"] if test else None, "paper_test_n": test["n"] if test else None})
    cursor = now
    for row in sorted(out, key=lambda r: r["state"] != "running"):  # stable: queue order otherwise
        if row["state"] != "done":
            cursor = cursor + dt.timedelta(hours=row["left_h"])
            row["finish"] = cursor
    return out


def main() -> int:
    table = rows()
    fmt = lambda t: t.strftime("%a %d %b %H:%M") if t else "-"
    print(f"{'run':30s} {'state':8s} {'steps':>15s} {'%':>4s}  {'left':>6s}  {'finish (est.)':17s} {'best val px':>11s} {'paper val mm':>12s} {'paper test mm':>14s}")
    for r in table:
        best = "" if r["best_val_px"] is None else f"{r['best_val_px']:.3f}"
        paper = "" if r["paper_val_mm"] is None else f"{r['paper_val_mm']:.2f}"
        test = "" if r["paper_test_mm"] is None else f"{r['paper_test_mm']:.2f} (n={r['paper_test_n']})"
        print(f"{r['run']:30s} {r['state']:8s} {r['step']:>7,d}/{r['total']:>7,d} {r['percent']:4.0f}  "
              f"{r['left_h']:5.1f}h  {fmt(r['finish']):17s} {best:>11s} {paper:>12s} {test:>14s}")
    done = sum(r["state"] == "done" for r in table)
    print(f"\n{done}/{len(table)} done; queue ends {fmt(table[-1]['finish'])} (now {fmt(dt.datetime.now())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
