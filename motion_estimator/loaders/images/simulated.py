"""Reader for ``realistic_dataset_v1`` -- simulated echo with exact GT motion.

Burman et al., KU Leuven: CAMUS LV geometries deformed by CircAdapt strain
patterns, imaged through the COLE convolution simulator. The reason it matters
here is the ground truth: the myocardium is a cloud of scattering sites whose
positions are stored **for every frame**, so correspondence between any two
frames is exact and dense, with no tracker in the loop. That is the supervision
LoFTR needs and the in-vivo cohort cannot provide.

Facts established by opening the files, not from the readme:

* **MATLAB v7.3, so HDF5.** ``scipy.io.loadmat`` raises; this module uses
  ``h5py``. Cell arrays come back as object arrays of HDF5 references, which is
  why ``scat_x`` needs a second dereference per frame.
* **Files are ~2.4 GB, and the cohort is 1.5 TB.** Most of that is
  ``RF_image``, which nothing here needs -- :func:`read_simulated` never
  touches it. Frames and scatterers together are ~5% of a file.
* **Axis order is reversed by HDF5.** ``cart_image`` reads as
  ``(n_frames, Nx, Ny)``; MATLAB's own ``(Ny, Nx, n_frames)`` comes back by
  transposing each frame, which is what this module returns.
* **Outside the sector is NaN**, not zero -- so the sector mask is free
  (``numpy.isfinite``), and any normalisation has to be NaN-aware.
* **``myocardial_scat_ind`` is ``1..79644``**: the myocardial scatterers are the
  leading block of the scatter map, the same set in every frame, stored 1-based.
  The remaining ~1.48 M sites are background clutter whose motion is not a
  meaningful target.
* **Pixel conversion** is the one thing the bundled MATLAB script states
  outright, and it is not symmetric::

      x_px = scat_x / dx + 1
      y_px = scat_y / dy + Ny / 2 + 1

  Verified visually: the converted points sit on the myocardium.
* **The simulated sector points left** (apex at low column), where the clinical
  view has the apex at the top -- the bundled script papers over this with
  ``view([90 90])`` and ``XDir reverse`` at plot time. ``orientation="clinical"``
  rotates the frames and the scatterers together so the geometry matches the
  in-vivo pipeline.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_SIM_ROOT",
    "SIM_TYPES",
    "SimClipRef",
    "SimulatedClip",
    "index_simulated",
    "simulated_root",
    "read_simulated",
    "read_simulated_meta",
]

DEFAULT_SIM_ROOT = Path(r"F:\realistic_dataset_v1")

# The readme lists four; which ones a patient actually has varies, so the
# inventory reports what is on disk rather than assuming the full set.
SIM_TYPES = (
    "regional_simulations",
    "global_simulations",
    "global_simulations1",
    "global_simulations2",
)

_FILENAME = re.compile(
    r"COLE_patient_(?P<patient>\d+)_CH_(?P<chambers>\d+)_sim_no_(?P<sim_no>\d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SimClipRef:
    """One simulated recording, identified the way the tree names it."""

    patient: int
    sim_type: str
    view: str
    sim_no: int
    path: Path

    @property
    def key(self) -> str:
        return f"Pat_{self.patient}/{self.sim_type}/{self.view}/sim{self.sim_no}"

    @property
    def group_id(self) -> str:
        """What cross-validation folds must be grouped on.

        One CAMUS patient supplies the geometry for every simulation derived
        from it -- both views, every ``sim_type``, every repeat. Splitting on
        anything finer puts the same ventricle on both sides of the split.
        """
        return f"Pat_{self.patient}"


def simulated_root(root: str | Path | None = None) -> Path:
    """Resolve the dataset root, honouring ``$MOTION_ESTIMATOR_SIM_ROOT``."""
    if root is None:
        root = os.environ.get("MOTION_ESTIMATOR_SIM_ROOT") or DEFAULT_SIM_ROOT
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"simulated dataset root {root} is not a directory. Set "
            "MOTION_ESTIMATOR_SIM_ROOT if the drive letter changed."
        )
    return root


def index_simulated(root: str | Path | None = None, sim_types=None) -> list[SimClipRef]:
    """Every recording under *root*, sorted by patient then filename."""
    root = simulated_root(root)
    wanted = set(sim_types) if sim_types else None

    refs: list[SimClipRef] = []
    for patient_dir in sorted(root.glob("Pat_*")):
        if not patient_dir.is_dir():
            continue
        try:
            patient = int(patient_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        for sim_dir in sorted(patient_dir.iterdir()):
            if not sim_dir.is_dir():
                continue
            if wanted is not None and sim_dir.name not in wanted:
                continue
            for path in sorted(sim_dir.glob("COLE_patient_*.mat")):
                match = _FILENAME.search(path.stem)
                if not match:
                    continue
                refs.append(
                    SimClipRef(
                        patient=patient,
                        sim_type=sim_dir.name,
                        view=f"{int(match.group('chambers'))}ch",
                        sim_no=int(match.group("sim_no")),
                        path=path,
                    )
                )
    return sorted(refs, key=lambda r: (r.patient, r.sim_type, r.view, r.sim_no))


@dataclass
class SimulatedClip:
    """One recording: frames, exact scatterer tracks, and the timing.

    ``frames`` is ``(n_frames, rows, cols)`` with **NaN outside the sector**.
    ``scat_x`` / ``scat_y`` are ``(n_frames, n_scatterers)`` in **1-based
    pixels**, the convention the rest of this package uses, and index the same
    scatterer in every row -- that identity is the ground truth.

    ``ed_frame`` / ``es_frame`` are 1-based indices into the **full**
    recording, mapped from the CircAdapt event indices (which count model
    steps, not frames). They do not shift when *frames* selects a subset.
    """

    frames: np.ndarray
    scat_x: np.ndarray
    scat_y: np.ndarray
    pixel_size_mm: np.ndarray
    frame_interval_s: float
    ed_frame: int
    es_frame: int
    long_strain: np.ndarray | None
    rad_strain: np.ndarray | None
    ejection_fraction: float | None
    heart_rate: float | None
    orientation: str
    ref: SimClipRef | None = None

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_scatterers(self) -> int:
        return int(self.scat_x.shape[1])

    @property
    def sector_mask(self) -> np.ndarray:
        """True inside the imaged sector -- free, because outside it is NaN."""
        return np.isfinite(self.frames).all(axis=0)

    def displacement(self, source: int, target: int) -> np.ndarray:
        """``(n_scatterers, 2)`` exact displacement between two frames."""
        return np.stack(
            [
                self.scat_x[target] - self.scat_x[source],
                self.scat_y[target] - self.scat_y[source],
            ],
            axis=-1,
        )


def _first(dataset) -> float:
    return float(np.asarray(dataset[()]).ravel()[0])


def _deref(handle, cell, index):
    """One entry of a MATLAB cell array stored as HDF5 references."""
    return np.asarray(handle[cell[index, 0]][()]).ravel()


def read_simulated_meta(path: str | Path) -> dict:
    """Shapes, timing and indices only -- no pixel or scatterer data read.

    Cheap enough to run over the whole 1.5 TB cohort, which is what the
    inventory does.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        motion = handle["KMsim/COLE/GTMotion"]
        image = handle["KMsim/COLE/cart_image"]
        n_frames, n_cols, n_rows = image.shape  # HDF5 reverses MATLAB's order
        events = handle["KMsim/CircAdapt/events"]
        more = handle.get("KMsim/CircAdapt/moreinfo")
        circ = handle["KMsim/CircAdapt"]

        n_scatterers = handle[motion["myocardial_scat_ind"][0, 0]].shape[0]
        return {
            "path": str(path),
            "n_frames": int(n_frames),
            "rows": int(n_rows),
            "cols": int(n_cols),
            "n_myocardial_scatterers": int(n_scatterers),
            "dx_mm": _first(motion["dx"]) * 1000.0,
            "dy_mm": _first(motion["dy"]) * 1000.0,
            "dt_s": _first(motion["dt"]),
            "n_mvc": _first(events["nMVC"]),
            "n_avc": _first(events["nAVC"]),
            "n_model_steps": int(_strain_length(circ)),
            "ejection_fraction": _first(more["EF"]) if more is not None else None,
            "heart_rate": _first(more["HR"]) if more is not None else None,
            "strain_fields": sorted(
                k for k in circ if k.endswith("Strain")
            ),
            "file_gb": path.stat().st_size / 1e9,
        }


def _strain_length(circ) -> int:
    for name in ("LongStrain", "LvfwLongStrain", "SpLongStrain"):
        if name in circ:
            return int(np.asarray(circ[name]).size)
    return 0


def _event_frames(circ, events, n_frames: int) -> tuple[int, int]:
    """Map CircAdapt event indices onto frame numbers, 1-based.

    The events index model steps (``nMVC``..``len(strain)``), and the recording
    resamples that span onto its own frames -- this is the same mapping the
    bundled MATLAB script performs before plotting.
    """
    first = _first(events["nMVC"])
    last = _strain_length(circ)
    es = _first(events["nAVC"])
    if last <= first or n_frames < 2:
        return 1, max(1, n_frames // 2)
    fraction = (es - first) / (last - first)
    es_frame = int(round(1 + fraction * (n_frames - 1)))
    return 1, int(np.clip(es_frame, 1, n_frames))


def read_simulated(
    path: str | Path,
    frames=None,
    max_scatterers: int | None = None,
    orientation: str = "clinical",
    dtype=np.float32,
    ref: SimClipRef | None = None,
) -> SimulatedClip:
    """Read one recording's frames and myocardial scatterer tracks.

    Parameters
    ----------
    frames:
        Which frames to read, as indices; ``None`` reads all. Reading a subset
        is much cheaper than reading the clip -- each frame is ~10 MB.
    max_scatterers:
        Keep an evenly spaced subset of the ~79 k myocardial scatterers.
        Evenly spaced rather than random so repeated reads agree, and so the
        subset stays spread over the wall instead of clumping.
    orientation:
        ``"clinical"`` rotates the sector apex-up, to match the in-vivo
        pipeline; ``"native"`` leaves it as simulated, pointing left.
    """
    import h5py

    if orientation not in ("clinical", "native"):
        raise ValueError(f"orientation must be clinical or native, got {orientation!r}")

    path = Path(path)
    with h5py.File(path, "r") as handle:
        motion = handle["KMsim/COLE/GTMotion"]
        image = handle["KMsim/COLE/cart_image"]
        circ = handle["KMsim/CircAdapt"]
        n_total, n_cols, n_rows = image.shape

        indices = range(n_total) if frames is None else [int(f) for f in frames]
        indices = list(indices)

        dx_mm = _first(motion["dx"]) * 1000.0
        dy_mm = _first(motion["dy"]) * 1000.0

        # 1-based, and the same leading block of the scatter map in every
        # frame -- so it is read once.
        keep = _deref(handle, motion["myocardial_scat_ind"], 0).astype(np.int64) - 1
        if max_scatterers is not None and keep.size > max_scatterers:
            keep = keep[np.linspace(0, keep.size - 1, max_scatterers).astype(int)]

        stack, xs, ys = [], [], []
        for index in indices:
            # HDF5 hands back (cols, rows); MATLAB's own layout is the transpose.
            stack.append(np.asarray(image[index], dtype=dtype).T)
            xs.append(_deref(handle, motion["scat_x"], index)[keep] / dx_mm * 1000.0)
            ys.append(_deref(handle, motion["scat_y"], index)[keep] / dy_mm * 1000.0)

        long_strain = _optional_strain(circ, ("LongStrain", "LvfwLongStrain"))
        rad_strain = _optional_strain(circ, ("RadStrain", "LvfwRadStrain"))
        more = handle.get("KMsim/CircAdapt/moreinfo")
        # Mapped onto the whole recording, never onto a frame subset: the
        # event indices describe the cardiac cycle, and asking for three
        # frames must not move end-systole to frame 2.
        ed_frame, es_frame = _event_frames(circ, circ["events"], int(n_total))
        ejection = _first(more["EF"]) if more is not None else None
        rate = _first(more["HR"]) if more is not None else None

    out_frames = np.stack(stack)
    # The MATLAB conversion, applied after the metre->pixel scaling above.
    scat_x = np.stack(xs) + 1.0
    scat_y = np.stack(ys) + n_rows / 2.0 + 1.0
    pixel_size = np.array([dx_mm, dy_mm])

    if orientation == "clinical":
        out_frames, scat_x, scat_y, pixel_size = _to_clinical(
            out_frames, scat_x, scat_y, pixel_size
        )

    return SimulatedClip(
        frames=out_frames,
        scat_x=scat_x,
        scat_y=scat_y,
        pixel_size_mm=pixel_size,
        frame_interval_s=_frame_interval(path),
        ed_frame=ed_frame,
        es_frame=es_frame,
        long_strain=long_strain,
        rad_strain=rad_strain,
        ejection_fraction=ejection,
        heart_rate=rate,
        orientation=orientation,
        ref=ref,
    )


def _frame_interval(path: Path) -> float:
    import h5py

    with h5py.File(path, "r") as handle:
        return _first(handle["KMsim/COLE/GTMotion"]["dt"])


def _optional_strain(circ, names) -> np.ndarray | None:
    for name in names:
        if name in circ:
            return np.asarray(circ[name]).ravel().astype(float)
    return None


def _to_clinical(frames, scat_x, scat_y, pixel_size):
    """Rotate the sector apex-up, moving the scatterers with it.

    The simulated sector points left. A clockwise quarter turn sends
    ``(row, col) -> (col, n_rows - 1 - row)``, so the points transform with the
    pixels rather than being re-derived, and the pixel sizes swap with the axes.
    """
    n_rows = frames.shape[1]
    rotated = np.rot90(frames, k=-1, axes=(1, 2))
    # 1-based in, 1-based out.
    new_x = n_rows - scat_y + 1.0
    new_y = scat_x
    return np.ascontiguousarray(rotated), new_x, new_y, pixel_size[::-1].copy()
