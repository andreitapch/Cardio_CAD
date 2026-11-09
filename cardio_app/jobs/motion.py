import sys

from cardio_app.jobs import runner

WORKER = "motion_estimator.viewer.worker"


def start(label, models, clips, flows):
    folder = runner.workspace()
    commands, files = [], []
    for model in dict.fromkeys(m for m in models if m):
        progress = folder / f"{model}.json"
        command = [sys.executable, "-m", WORKER, "--model", model, "--progress", str(progress)]
        command += ["--all"] if clips is None else ["--clips", *clips]
        if flows:
            command.append("--flows")
        commands.append(command)
        files.append(progress)
    return runner.start(label, commands, files, folder)
