"""Fixtures for the motion_estimator data readers.

Most tests here run without the dataset drive mounted, against MAT files this
module synthesises with ``scipy.io.savemat``. That is deliberate: the readers
have to be verifiable while the external drive is unplugged.

Two fixtures do need real data and skip when it is absent:

``demo_srd``
    A Speqle ``.srd`` shipped with the Speqle3D source tree on the local disk.
    Proves the reader against a file this project did not write.
``dataset_root``
    The intervendor cohort itself. Honours ``$MOTION_ESTIMATOR_DATA_ROOT``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from motion_estimator.loaders.dataset_index import dataset_root as resolve_root

DEMO_SRD = Path(
    r"D:\PHD\Development\Matlab\speqle3d\Speqle3D_source\DemoFiles\Clinical2Dt_2ch.srd"
)

# Pinned from the real files, so a reader regression shows up as a value change
# rather than only as a crash.
IV02_CYCLE_LIMITS = {"2ch": (51, 94), "4ch": (50, 93)}
IV02_GLS_ES = {"2ch": -10.6, "4ch": -8.8}
IV02_N_CONTOUR_POINTS = 61
IV02_N_SAMPLES = 159


@pytest.fixture(scope="session")
def demo_srd() -> Path:
    if not DEMO_SRD.is_file():
        pytest.skip(f"Speqle demo clip not available at {DEMO_SRD}")
    return DEMO_SRD


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    root = resolve_root()
    if not root.is_dir():
        pytest.skip(
            f"dataset root {root} not mounted; set $MOTION_ESTIMATOR_DATA_ROOT"
        )
    return root


@pytest.fixture(scope="session")
def iv02_dir(dataset_root: Path) -> Path:
    path = dataset_root / "Retest_dataset" / "Data" / "IV02"
    if not path.is_dir():
        pytest.skip(f"{path} not present")
    return path


def make_clip_mat(path: Path, tissue: np.ndarray, voxelsize=(5e-4, 6e-4, 1e-3)) -> Path:
    """Write a minimal but structurally faithful Speqle ``tissue3`` clip."""
    sio.savemat(
        path,
        {
            "USInfo": {
                "srd_format": "V1.3",
                "type": "GEVivid7",
                "mode": "tissue3D_sc",
                "patientsname": "anonymous",
                "patientid": "0",
                "gridsize": np.asarray(tissue.shape, dtype=np.uint16).reshape(1, -1),
                "voxelsize": np.asarray(voxelsize, dtype=float).reshape(1, 3),
                "frames": np.uint8(1),
            },
            "USData": {
                "tissue3": tissue,
                "ecg": np.zeros((1, 10)),
            },
        },
    )
    return path


def make_scar_struct_mat(
    path: Path,
    views=("2ch", "4ch"),
    limits=((51, 94), (50, 93)),
    n_points: int = IV02_N_CONTOUR_POINTS,
    n_samples: int = IV02_N_SAMPLES,
) -> Path:
    """Write a ``scar_struct.mat`` with the real field names and shapes."""
    scar = {}
    for view, (first, last) in zip(views, limits):
        segment_sizes = _split_sizes(n_samples, 6)
        segment_definition = np.empty((1, 6), dtype=object)
        start = 1
        for index, size in enumerate(segment_sizes):
            # 1-based indices, as MATLAB stores them.
            segment_definition[0, index] = np.arange(
                start, start + size, dtype=np.uint8
            ).reshape(1, -1)
            start += size

        scar[f"info_{view}"] = {
            "cycle_frame_limits": np.asarray(
                [[first, last]], dtype=np.uint8
            ),
            "contour": {
                "surf_endo": np.zeros((2, n_points)),
                "surf_epi": np.ones((2, n_points)),
                "surf_endo_ind": np.zeros((2, n_points)),
                "surf_epi_ind": np.ones((2, n_points)),
                "seg_definition": np.ones((n_points, 1), dtype=np.uint8),
            },
            "contour_sampled": {
                "Endo_sample_x": np.arange(n_samples, dtype=float).reshape(1, -1),
                "Endo_sample_y": np.arange(n_samples, dtype=float).reshape(1, -1) * 2,
                "Epi_sample_x": np.arange(n_samples, dtype=float).reshape(1, -1) + 5,
                "Epi_sample_y": np.arange(n_samples, dtype=float).reshape(1, -1) * 3,
                "segment_definition": segment_definition,
            },
        }
    sio.savemat(path, {"scar_struct": scar})
    return path


def make_gt_strain_mat(path: Path, views=("2ch", "4ch")) -> Path:
    """Write a ``GT_STRAIN`` file with IV02's real 2ch/4ch values."""
    real = {
        "2ch": {
            "peak_systolic": [-15.42, -9.26, -2.28, 3.59, -5.13, -9.70],
            "end_systolic": [-15.42, -9.26, -2.28, -1.06, -5.13, -9.70],
            "post_systolic": [-15.42, -10.76, -11.40, -6.37, -7.41, -11.50],
            "gls": -10.6,
        },
        "4ch": {
            "peak_systolic": [-9.07, -11.36, 5.08, 14.17, 2.39, -9.24],
            "end_systolic": [-9.07, -11.36, -0.78, 13.87, 0.05, -9.24],
            "post_systolic": [-9.67, -13.66, -7.96, -9.27, -6.39, -11.98],
            "gls": -8.8,
        },
    }
    strain = {}
    for view in views:
        values = real[view]
        for phase in ("peak_systolic", "end_systolic", "post_systolic"):
            strain[f"segm_{view}_{phase}_strain"] = np.asarray(
                values[phase], dtype=float
            ).reshape(1, 6)
        strain[f"GLS_{view}_ES"] = np.asarray([[values["gls"]]], dtype=float)
    sio.savemat(path, {"strain": strain})
    return path


def _split_sizes(total: int, parts: int) -> list[int]:
    """Contiguous partition sizes, remainder spread over the leading parts."""
    base, extra = divmod(total, parts)
    return [base + (1 if index < extra else 0) for index in range(parts)]


@pytest.fixture
def fake_cohort(tmp_path: Path) -> Path:
    """A miniature two-folder cohort exercising the awkward real-world cases.

    Mirrors what the actual tree does: ``IV02`` has a ``.srd`` for 2ch but only
    an ``.h5`` for 4ch, ``IV03`` has no ground truth at all, and the same
    patient ids recur across both dataset folders while being different people.
    """
    tissue = np.zeros((4, 3, 100), dtype=np.uint8)

    for dataset in ("Retest_dataset", "Test_dataset"):
        for patient in ("IV02", "IV03"):
            patient_dir = tmp_path / dataset / "Data" / patient
            (patient_dir / "Analysis").mkdir(parents=True)
            make_scar_struct_mat(patient_dir / "Analysis" / "scar_struct.mat")

            for view in ("2ch", "4ch"):
                (patient_dir / view).mkdir()
                make_clip_mat(patient_dir / view / "data.mat", tissue)

            make_clip_mat(patient_dir / f"{patient}_2ch.srd", tissue)
            (patient_dir / f"{patient}_4ch.h5").write_bytes(b"")

            if patient != "IV03":
                gt_dir = tmp_path / dataset / "GT_STRAIN" / "GE_mat"
                gt_dir.mkdir(parents=True, exist_ok=True)
                make_gt_strain_mat(gt_dir / f"GE_strain{patient}.mat")

    return tmp_path
