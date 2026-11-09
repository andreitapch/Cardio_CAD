import os
from pathlib import Path

from pattern_recognition.utils.parameters_dics import dbConParameters

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
JOBS = ARTIFACTS / "jobs"
REPORTS = REPO / "reports"
STUDY_TYPE = "BL"
DB = {
    "driver": os.environ.get("CARDIO_DB_DRIVER", dbConParameters["driver"]),
    "server": os.environ.get("CARDIO_DB_SERVER", dbConParameters["server"]),
    "database": os.environ.get("CARDIO_DB_NAME", dbConParameters["database"]),
}
