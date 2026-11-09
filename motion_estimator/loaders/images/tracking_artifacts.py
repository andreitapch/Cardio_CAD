"""Readers for the per-clip ``Tracking/`` and ``srd/analysis.mat`` artifacts.

These sit beside every clip and were not part of the original port, so nothing
downstream depended on them being right. What they hold, established by opening
the files rather than from documentation:

``<view>/Tracking/data<k>.mhd`` + ``.raw``
    The cycle as per-frame images. **The header lies**: it says
    ``DimSize = 500 500`` at ``MET_DOUBLE``, which is 2 MB, but each file is
    4 MB. The array is the frame written *twice*, and each half matches
    ``read_clip(...).tissue[..., cycle_start - 1 + k]`` to a correlation of
    1.0000 -- so these are in the **raw** ``tissue3`` orientation, not the
    clinical view, and carry no new pixel data. Useful only to know what
    geometry the elastix registrations below were computed in.

``<view>/Tracking/Param*/TransformParameters<k>-<k+1>.txt``
    An ITK B-spline transform per consecutive frame pair -- a dense
    deformation field for the whole cycle, already computed. 116/120 clips
    have them. This is a registration-based tracking baseline to compare
    against, and the coordinates are the raw 500x500 grid at the ``.mhd``
    spacing of 0.1 per pixel.

``<view>/Tracking/Results1e2/``
    ``strain_mat.mat`` (``ps_strain``, ``pss_strain``, ``strain_curves``,
    ``ES_strains``, ``PSI``), ``GL_average_segs.mat`` (``global_endo``,
    ``endo_ES_val``) and three ``exclude*.mat`` files -- per-segment
    quality flags. Worth reading before blaming a tracker for a clip that was
    already marked unusable.

``<view>/srd/analysis.mat``
    ``analysis_info.BEASData`` is ``(1, n_frames)``: a BEAS segmentation per
    frame, each carrying a 48-point contour in ``SurfInfo.SurfX``/``SurfY``,
    a 61-point ``MITTx``/``MITTy``, and the 6 ``userPoints`` landmarks.

    **Its coordinate frame is unresolved.** It is not the 500x500 clinical
    grid: ``VoxelSize`` is isotropic 0.26052 mm against the clip's
    ``(0.3172, 0.26052)``, and ``xSize`` is 609 -- which is
    ``500 * 0.3172 / 0.26052``, so BEAS ran on an isotropically resampled
    volume (``srd/Beas_iso.srd``). Resampling alone does not place the
    contour: every combination of axis swap, flip and isotropic rescale leaves
    a mean nearest-point distance of 34 px from the verified ``scar_struct``
    contour, and the object also carries a ``Heart2World`` rotation, ``cx``
    /``cy`` centre and ``PadData = 50``, none of which close the gap either.
    :func:`read_beas_contours` therefore returns the values **as stored**, with
    ``frame_resolved = False``, and refuses to guess. Settling it means
    tracing the 2D BEAS surface construction in
    ``speqle3d/Speqle3D_source/Speqle_3D/BEAS_toolbox``; the 3D
    ``BEAS_UpdateSurface.m`` formula does not reproduce these arrays.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import _matio

__all__ = [
    "BeasFrame",
    "BeasContours",
    "ItkBSpline",
    "TrackingResults",
    "read_beas_contours",
    "read_itk_bspline",
    "read_tracking_frames",
    "read_tracking_results",
    "read_vtk_points",
]


# --------------------------------------------------------------------------
# Tracking/data<k>.raw
# --------------------------------------------------------------------------


def read_tracking_frames(directory: str | Path, shape: tuple[int, int] = (500, 500)):
    """Every ``data<k>.raw`` in *directory*, as ``(n_frames, rows, cols)``.

    The duplicate half is dropped -- see the module docstring. Frames come back
    in the **raw** ``tissue3`` orientation, matching what the elastix
    transforms were computed on, *not* the clinical view.
    """
    directory = Path(directory)
    paths = sorted(
        directory.glob("data*.raw"),
        key=lambda p: int(re.search(r"data(\d+)", p.stem).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"no data*.raw under {directory}")

    rows, cols = shape
    expected = rows * cols
    frames = []
    for path in paths:
        values = np.fromfile(path, dtype="<f8")
        if values.size == expected:
            frames.append(values.reshape(rows, cols))
        elif values.size == 2 * expected:
            # Written twice; the halves are identical, so take the first.
            frames.append(values[:expected].reshape(rows, cols))
        else:
            raise ValueError(
                f"{path.name}: {values.size} doubles is neither {expected} nor "
                f"{2 * expected} -- the .mhd DimSize is not to be trusted here"
            )
    return np.stack(frames)


def _frame_index(path: Path) -> tuple[int, int]:
    match = re.search(r"(\d+)-(\d+)", path.stem)
    if not match:
        raise ValueError(f"{path.name}: no <k>-<k+1> frame pair in the name")
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------
# Tracking/Param*/TransformParameters<k>-<k+1>.txt
# --------------------------------------------------------------------------


@dataclass
class ItkBSpline:
    """One ITK ``AdvancedBSplineDeformableTransform``, as stored on disk.

    ``coefficients`` is ``(2, ny, nx)`` -- ITK stores every x-component first,
    then every y-component, each in Fortran order over the control grid, so
    the split and the reshape are both part of the format rather than a
    convention chosen here.

    Coordinates are **physical**, not pixels: the grid origin and spacing come
    from ``FixedParameters`` and are in the units the ``.mhd`` declares, where
    ``ElementSpacing = 0.1``. Use :meth:`transform_points` with
    *image_spacing* to work in pixels.
    """

    coefficients: np.ndarray
    grid_size: tuple[int, int]
    grid_origin: tuple[float, float]
    grid_spacing: tuple[float, float]
    grid_direction: np.ndarray
    transform: str
    frames: tuple[int, int] | None = None
    source: Path | None = None

    def transform_points(self, x, y, image_spacing: float | tuple[float, float] = 0.1):
        """Displace 0-based pixel coordinates by this transform.

        Returns ``(moved_x, moved_y)`` in pixels. *image_spacing* converts
        pixels to the physical frame the transform lives in; the ``.mhd``
        files beside these transforms all declare ``0.1``.
        """
        spacing = np.broadcast_to(np.asarray(image_spacing, dtype=float), (2,))
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        physical = np.stack([x * spacing[0], y * spacing[1]])
        moved = physical + self._displacement(physical)
        return moved[0] / spacing[0], moved[1] / spacing[1]

    def _displacement(self, physical: np.ndarray) -> np.ndarray:
        """Cubic B-spline tensor product over the 4x4 supporting control points."""
        origin = np.asarray(self.grid_origin, dtype=float)[:, None]
        spacing = np.asarray(self.grid_spacing, dtype=float)[:, None]
        flat = physical.reshape(2, -1)

        # Continuous index into the control grid, and the 4-point support that
        # a cubic B-spline reads: floor(index) - 1 .. floor(index) + 2.
        index = (flat - origin) / spacing
        base = np.floor(index).astype(int) - 1
        frac = index - np.floor(index)

        weights = np.stack([_cubic_bspline(frac + 1.0 - k) for k in range(4)])
        n_x, n_y = self.grid_size

        out = np.zeros_like(flat)
        for offset_y in range(4):
            row = base[1] + offset_y
            valid_row = (row >= 0) & (row < n_y)
            for offset_x in range(4):
                col = base[0] + offset_x
                valid = valid_row & (col >= 0) & (col < n_x)
                if not valid.any():
                    continue
                weight = weights[offset_x, 0] * weights[offset_y, 1]
                contribution = np.where(
                    valid,
                    self.coefficients[:, np.clip(row, 0, n_y - 1), np.clip(col, 0, n_x - 1)]
                    * weight,
                    0.0,
                )
                out += contribution
        return out.reshape(physical.shape)


def _cubic_bspline(t: np.ndarray) -> np.ndarray:
    """The uniform cubic B-spline basis, evaluated at *t*."""
    t = np.abs(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    near = t < 1.0
    far = (t >= 1.0) & (t < 2.0)
    out[near] = (4.0 - 6.0 * t[near] ** 2 + 3.0 * t[near] ** 3) / 6.0
    out[far] = (2.0 - t[far]) ** 3 / 6.0
    return out


def _read_numbers(text: str, key: str) -> np.ndarray:
    match = re.search(rf"^{key}:\s*(.*)$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"no '{key}:' line")
    return np.fromstring(match.group(1), sep=" ")


def read_itk_bspline(path: str | Path) -> ItkBSpline:
    """Parse one ``TransformParameters<k>-<k+1>.txt``."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    name = re.search(r"^Transform:\s*(\S+)", text, flags=re.MULTILINE)
    transform = name.group(1) if name else "unknown"
    if "BSpline" not in transform:
        raise ValueError(
            f"{path.name}: only B-spline transforms are handled, got {transform!r}"
        )

    parameters = _read_numbers(text, "Parameters")
    fixed = _read_numbers(text, "FixedParameters")
    if fixed.size != 10:
        raise ValueError(
            f"{path.name}: expected 10 FixedParameters for a 2-D B-spline, "
            f"got {fixed.size}"
        )

    grid_size = (int(fixed[0]), int(fixed[1]))
    grid_origin = (float(fixed[2]), float(fixed[3]))
    grid_spacing = (float(fixed[4]), float(fixed[5]))
    grid_direction = fixed[6:10].reshape(2, 2)

    n_control = grid_size[0] * grid_size[1]
    if parameters.size != 2 * n_control:
        raise ValueError(
            f"{path.name}: {parameters.size} parameters for a "
            f"{grid_size[0]}x{grid_size[1]} grid; expected {2 * n_control}"
        )
    # x-components then y-components, each Fortran-ordered over (nx, ny).
    coefficients = np.stack(
        [
            parameters[:n_control].reshape(grid_size[::-1]),
            parameters[n_control:].reshape(grid_size[::-1]),
        ]
    )

    try:
        frames = _frame_index(path)
    except ValueError:
        frames = None

    return ItkBSpline(
        coefficients=coefficients,
        grid_size=grid_size,
        grid_origin=grid_origin,
        grid_spacing=grid_spacing,
        grid_direction=grid_direction,
        transform=transform,
        frames=frames,
        source=path,
    )


def read_transform_sequence(directory: str | Path) -> list[ItkBSpline]:
    """Every consecutive-pair transform in a ``Param*`` directory, in order."""
    directory = Path(directory)
    paths = sorted(directory.glob("TransformParameters*.txt"), key=_frame_index)
    if not paths:
        raise FileNotFoundError(f"no TransformParameters*.txt under {directory}")
    return [read_itk_bspline(path) for path in paths]


# --------------------------------------------------------------------------
# Tracking/Results1e2/
# --------------------------------------------------------------------------


@dataclass
class TrackingResults:
    """What the registration pipeline derived, and what it excluded.

    ``exclude_segments`` and ``exclude_total`` are per-segment flags;
    ``segment_quality`` is a per-segment score. A clip already marked unusable
    here should not be counted against a new tracker, which is the point of
    reading them.
    """

    strain_curves: np.ndarray | None
    peak_systolic: np.ndarray | None
    post_systolic: np.ndarray | None
    es_strains: np.ndarray | None
    psi: np.ndarray | None
    global_endo: np.ndarray | None
    endo_es_value: float | None
    exclude_segments: np.ndarray | None
    exclude_total: np.ndarray | None
    segment_quality: np.ndarray | None
    source: Path | None = None

    # Deliberately no ``is_excluded`` helper. The polarity of these flags is
    # not established -- IV02 4ch reads ``exclude_seg = [1 1 1 1 1 1]``, and a
    # clip with every segment excluded is not a plausible reading of a file
    # that exists for all of them, so 1 more likely means "keep". Until that is
    # settled against the MATLAB that wrote them, callers get the raw arrays
    # and have to decide, rather than a helper that quietly picks a convention.


def _optional(struct, name: str):
    try:
        if not _matio.has_field(struct, name):
            return None
    except Exception:
        return None
    value = _matio.field(struct, name)
    return None if value is None else np.asarray(value)


def _single_variable(path: Path, name: str):
    if not path.is_file():
        return None
    variables = _matio.load_mat(path)
    for key, value in variables.items():
        if not key.startswith("__") and (name is None or key == name):
            return value
    return None


def read_tracking_results(directory: str | Path) -> TrackingResults:
    """Read a ``Results*`` directory -- curves, ES values and exclusions."""
    directory = Path(directory)
    strain = _single_variable(directory / "strain_mat.mat", "strain_mat")
    segs = _single_variable(directory / "GL_average_segs.mat", "GL_average_segs")

    endo_es = _optional(segs, "endo_ES_val") if segs is not None else None
    if endo_es is not None and endo_es.size == 1:
        endo_es = float(endo_es.ravel()[0])
    elif endo_es is not None:
        endo_es = None

    def flags(filename: str, field: str):
        value = _single_variable(directory / filename, None)
        return None if value is None else np.asarray(value).ravel()

    return TrackingResults(
        strain_curves=_optional(strain, "strain_curves") if strain is not None else None,
        peak_systolic=_optional(strain, "ps_strain") if strain is not None else None,
        post_systolic=_optional(strain, "pss_strain") if strain is not None else None,
        es_strains=_optional(strain, "ES_strains") if strain is not None else None,
        psi=_optional(strain, "PSI") if strain is not None else None,
        global_endo=_optional(segs, "global_endo") if segs is not None else None,
        endo_es_value=endo_es,
        exclude_segments=flags("exclude.mat", "exclude_seg"),
        exclude_total=flags("exclude_final.mat", "exclude_total"),
        segment_quality=flags("exclude_quality.mat", "exclude_seg"),
        source=directory,
    )


# --------------------------------------------------------------------------
# srd/analysis.mat -- BEAS
# --------------------------------------------------------------------------


@dataclass
class BeasFrame:
    """One frame's BEAS segmentation, in BEAS's own coordinates.

    ``surf_x`` / ``surf_y`` is the 48-point contour, ``mitt_x`` / ``mitt_y``
    the 61-point one BEAS carries alongside it (not the same sampling as the
    ``MITT_uni`` results, which have 72). ``user_points`` are the manual
    landmarks the operator placed.
    """

    surf_x: np.ndarray
    surf_y: np.ndarray
    mitt_x: np.ndarray
    mitt_y: np.ndarray
    user_points: np.ndarray
    centre: tuple[float, float]
    voxel_size_mm: np.ndarray
    grid_size: tuple[int, int]
    heart_to_world: np.ndarray
    pad: float


@dataclass
class BeasContours:
    """Every frame's BEAS segmentation for one clip.

    ``frame_resolved`` is **False**: these coordinates have not been mapped
    onto the clinical view, and nothing here should be plotted over a clip or
    compared with MITT until it has. See the module docstring for what was
    tried and why it was left unresolved rather than approximated.
    """

    frames: list[BeasFrame]
    frame_resolved: bool = False
    source: Path | None = None

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def surf_stack(self) -> tuple[np.ndarray, np.ndarray]:
        """``(n_frames, n_points)`` for x and y -- still unmapped."""
        return (
            np.stack([f.surf_x for f in self.frames]),
            np.stack([f.surf_y for f in self.frames]),
        )


def _beas_object(entry):
    """The BEAS object for one frame, or None where the frame has none.

    Not every frame carries a segmentation -- unsegmented frames are stored as
    an empty array rather than an empty struct, so the field access itself
    raises rather than returning something empty.
    """
    entry = np.asarray(entry)
    if entry.size == 0 or entry.dtype.names is None:
        return None
    try:
        meshes = np.asarray(_matio.field(entry, "meshesBEAS"))
    except (TypeError, KeyError, ValueError):
        return None
    if meshes.size == 0:
        return None
    candidate = np.asarray(meshes[0, 0])
    if candidate.dtype.names is None or "BEAS_object" not in candidate.dtype.names:
        return None
    return np.asarray(_matio.field(candidate, "BEAS_object"))


def read_beas_contours(path: str | Path) -> BeasContours:
    """Read ``srd/analysis.mat`` -- a contour per frame, in BEAS coordinates."""
    path = Path(path)
    info = _matio.load_mat(path)["analysis_info"]
    entries = np.asarray(_matio.field(info, "BEASData"))

    frames = []
    for index in range(entries.shape[-1]):
        obj = _beas_object(entries[0, index])
        if obj is None:
            continue
        surf = np.asarray(_matio.field(obj, "SurfInfo"))
        scalar = lambda name: float(np.asarray(_matio.field(obj, name)).ravel()[0])
        vector = lambda source, name: np.asarray(
            _matio.field(source, name), dtype=float
        ).ravel()

        frames.append(
            BeasFrame(
                surf_x=vector(surf, "SurfX"),
                surf_y=vector(surf, "SurfY"),
                mitt_x=vector(obj, "MITTx"),
                mitt_y=vector(obj, "MITTy"),
                user_points=np.asarray(_matio.field(obj, "userPoints"), dtype=float),
                centre=(scalar("cx"), scalar("cy")),
                voxel_size_mm=np.asarray(
                    _matio.field(obj, "VoxelSize"), dtype=float
                ).ravel() * 1000.0,
                grid_size=(int(scalar("xSize")), int(scalar("ySize"))),
                heart_to_world=np.asarray(
                    _matio.field(obj, "Heart2World"), dtype=float
                ),
                pad=scalar("PadData"),
            )
        )

    return BeasContours(frames=frames, frame_resolved=False, source=path)


# --------------------------------------------------------------------------
# Tracking/bench/contour<k>.vtk
# --------------------------------------------------------------------------


def read_vtk_points(path: str | Path) -> np.ndarray:
    """The ``POINTS`` block of an ASCII VTK POLYDATA file, as ``(n, 3)``."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("POINTS"):
            count = int(line.split()[1])
            values = np.fromstring(
                " ".join(lines[index + 1: index + 1 + count]), sep=" "
            )
            return values.reshape(count, 3)
    raise ValueError(f"{path.name}: no POINTS block")
