import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from cardio_app import settings


@dataclass
class Job:
    label: str
    commands: list
    progress_files: list
    log: Path
    meta: dict = field(default_factory=dict)
    process: subprocess.Popen | None = None
    index: int = 0
    started: float = field(default_factory=time.time)

    def running(self):
        return self.process is not None and self.process.poll() is None

    def failed(self):
        return self.process is not None and self.process.returncode not in (None, 0)

    def finished(self):
        return not self.running() and (self.failed() or self.index >= len(self.commands))

    def advance(self):
        if self.running() or self.failed() or self.index >= len(self.commands):
            return
        env = {**os.environ, "MKL_THREADING_LAYER": "TBB", "PYTHONPATH": str(settings.REPO)}
        with open(self.log, "a", encoding="utf-8") as handle:
            self.process = subprocess.Popen(self.commands[self.index], cwd=settings.REPO, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT)
        self.index += 1

    def cancel(self):
        if self.running():
            self.process.terminate()
        self.index = len(self.commands)

    def reports(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in self.progress_files if path.is_file()]

    def progress(self):
        parts = []
        for path in self.progress_files:
            if path.is_file():
                report = json.loads(path.read_text(encoding="utf-8"))
                parts.append(f"{report['name']}: {report['done']}/{report['total']}" + (" done" if report["state"] == "done" else ""))
            else:
                parts.append("waiting")
        suffix = " - failed, see log" if self.failed() else (" - finished" if self.finished() else "")
        return f"{self.label} - " + " | ".join(parts) + suffix


def workspace():
    folder = settings.JOBS / time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def start(label, commands, progress_files, folder, meta=None):
    job = Job(label, commands, progress_files, folder / "log.txt", meta or {})
    job.advance()
    return job


def report(path, name, done, total, state="running"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "done": done, "total": total, "state": state}), encoding="utf-8")
