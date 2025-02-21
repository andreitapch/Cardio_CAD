import json
import os
from pathlib import Path
import numpy as np

def load_parameters_from_json(filename: Path) -> dict:
    with open(filename, "r") as f:
        return json.load(f)

p = 'D:/PHD/Development/Python/Cardio_CAD/utils/SmoothAHAplot/src/aha/parameters'

AHA_FEATURES = load_parameters_from_json(os.path.join(p, 'aha_features.json'))
BIOMARKER_FEATURES = load_parameters_from_json(os.path.join(p, 'biomarker_features.json'))
PLOT_COMPONENTS = load_parameters_from_json(os.path.join(p, 'plot_components.json'))

ANGULAR_COORDINATES = np.linspace(0, 2 * np.pi, PLOT_COMPONENTS["resolution"][0])
RADIAL_COORDINATES = np.linspace(0, 1, PLOT_COMPONENTS["resolution"][1])