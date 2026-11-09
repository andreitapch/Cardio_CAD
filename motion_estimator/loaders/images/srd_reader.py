"""Read a Speqle ultrasound clip from ``.srd``, ``.mat`` or ``.h5``.

``.srd`` is not a bespoke binary format: ``pack_SRD_HDPULSE.m`` ends with a
plain ``save(write_file_name, 'USData', 'USInfo', '-mat')``, so an ``.srd`` is a
MAT-file with a different extension. Confirmed against
``speqle3d/Speqle3D_source/DemoFiles/Clinical2Dt_2ch.srd``, whose header reads
``MATLAB 5.0 MAT-file``.

Two payload layouts exist:

``tissue3``
    Scan-converted Cartesian data, with geometry in ``USInfo.gridsize`` and
    ``USInfo.voxelsize``. This is what the intervendor dataset uses, and what
    ``loadIntervendorData.m`` reads out of each ``data.mat``.
``tissue``
    The HDPulse acquisition layout: still in polar coordinates, with
    ``samples``/``beams``/``r0``/``dr``/``t0``/``dt`` instead. Loaded and
    reported, but scan conversion is not implemented, so
    :mod:`motion_estimator.loaders.images.image_prep` refuses it rather than producing a
    silently distorted image.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path

import numpy as np

from . import _matio

__all__ = ["UltrasoundClip", "read_clip", "read_h5_tissue"]

_INFO_TEXT_FIELDS = (
    "srd_format",
    "type",
    "mode",
    "patientsname",
    "patientid",
    "studydate",
    "studytime",
    "sopinstanceuid",
    "source",
    "description",
    "scanner_software_version",
    "scantype",
)


@dataclass
class UltrasoundClip:
    """One clip, with axes left in the order MATLAB stores them.

    ``tissue`` is ``(n_rows, n_cols, n_frames)`` for the 2D+t clips this
    package targets -- the raw acquisition order, *before*
    :func:`motion_estimator.loaders.images.image_prep.to_clinical_view` flips and
    transposes it. ``voxelsize`` stays in metres, as written; use
    :func:`~motion_estimator.loaders.images.image_prep.pixel_spacing_mm` for the
    millimetre spacing in clinical-view axis order.

    ``tissue3`` is also how Speqle stores volumetric data, so ``tissue`` may
    come back 4-D ``(x, y, z, n_frames)`` -- two of the bundled Speqle demo
    files are 3D+t. Nothing distinguishes a 2D+t clip from a single static
    volume in ``USInfo`` (both carry ``mode == 'tissue3D_sc'``), so for 3-D
    data axis 2 is taken to be the frame axis. That is correct for the
    intervendor dataset and for anything ``loadIntervendorData.m`` handled;
    :attr:`is_volumetric` flags the 4-D case rather than guessing at it.
    """

    tissue: np.ndarray
    layout: str
    voxelsize: np.ndarray | None
    gridsize: np.ndarray | None
    ecg: np.ndarray | None
    info: dict = _dc_field(default_factory=dict)
    source: Path | None = None

    @property
    def is_volumetric(self) -> bool:
        """True for 3D+t data, which the 2D strain pipeline cannot consume."""
        return self.tissue.ndim == 4

    @property
    def n_frames(self) -> int:
        """Length of the frame axis, which is the last axis in both layouts."""
        return int(self.tissue.shape[-1])

    def __repr__(self) -> str:  # keeps the pixel data out of the repr
        name = self.source.name if self.source else "<memory>"
        return (
            f"UltrasoundClip({name}, layout={self.layout!r}, "
            f"tissue={self.tissue.shape} {self.tissue.dtype}, "
            f"mode={self.info.get('mode', '')!r})"
        )


def read_clip(path: str | Path) -> UltrasoundClip:
    """Load an ``.srd`` or ``.mat`` clip.

    ``.h5`` is rejected, because it is not an equivalent export. The three
    files in an ``IV##`` directory are one acquisition at three different
    geometries -- their frame counts agree and nothing else does:

    ==================== ================================ =====================
    file                 shape                            geometry
    ==================== ================================ =====================
    ``<view>/data.mat``  ``(500, 500, n_frames)``         scan-converted
    ``IV##_<view>.srd``  ``(~214-256, 200, n_frames)``    scan-converted, coarse
    ``IV##_<view>.h5``   ``Tissue(n_frames, beams, samples)``  polar
    ==================== ================================ =====================

    The HDF5 export holds un-scan-converted polar data straight off the Vivid
    E9, with the frame axis first, so no transpose maps it onto the MAT files;
    turning it into an image needs scan conversion, exactly as the ``tissue``
    layout does. Use ``<view>/data.mat`` -- which is also the only export
    present for every view -- and :func:`read_h5_tissue` when the polar data or
    the HDF5 metadata is what you actually want.
    """
    path = Path(path)
    if path.suffix.lower() == ".h5":
        raise NotImplementedError(
            f"{path.name}: this export holds polar (beam, sample) data with the "
            "frame axis first, not a scan-converted image, and scan conversion "
            "is not implemented here. Read the view's data.mat instead, or "
            "read_h5_tissue() for the polar array and its scanner metadata."
        )

    variables = _matio.load_mat(path)
    missing = {"USInfo", "USData"} - set(variables)
    if missing:
        raise ValueError(
            f"{path.name}: not a Speqle clip, missing {sorted(missing)}; "
            f"found {sorted(variables)}"
        )

    us_info, us_data = variables["USInfo"], variables["USData"]
    data_fields = _matio.fields(us_data)

    if "tissue3" in data_fields:
        layout, tissue = "tissue3", _matio.field(us_data, "tissue3")
    elif "tissue" in data_fields:
        layout, tissue = "tissue", _matio.field(us_data, "tissue")
    else:
        raise ValueError(
            f"{path.name}: USData has neither 'tissue3' nor 'tissue'; "
            f"found {sorted(data_fields)}"
        )

    tissue = np.asarray(tissue)
    if tissue.ndim not in (3, 4):
        raise ValueError(
            f"{path.name}: expected a 3-D or 4-D {layout} array, "
            f"got shape {tissue.shape}"
        )

    return UltrasoundClip(
        tissue=tissue,
        layout=layout,
        voxelsize=_optional_numeric(us_info, "voxelsize"),
        gridsize=_optional_numeric(us_info, "gridsize"),
        ecg=_optional_numeric(us_data, "ecg"),
        info=_read_info(us_info),
        source=path,
    )


def _optional_numeric(struct, name: str) -> np.ndarray | None:
    if not _matio.has_field(struct, name):
        return None
    value = np.asarray(_matio.field(struct, name))
    if value.size == 0:
        return None
    return value.ravel() if name in ("voxelsize", "gridsize") else value


def _read_info(us_info) -> dict:
    """Everything from ``USInfo`` except the bulk arrays, as plain Python."""
    info: dict = {}
    for name in _matio.fields(us_info):
        if name in ("preview", "voxelsize", "gridsize"):
            continue
        value = _matio.field(us_info, name)
        if name in _INFO_TEXT_FIELDS:
            info[name] = _matio.text(value)
            continue
        array = np.asarray(value)
        if array.size == 1 and array.dtype.kind in "iufb":
            info[name] = _matio.scalar(array)
        elif array.size == 0:
            info[name] = None
        else:
            info[name] = array
    return info


def read_h5_tissue(path: str | Path) -> tuple[np.ndarray, dict]:
    """Polar tissue array and metadata from an ``IV##_<view>.h5`` export.

    Returned **exactly as stored**: ``(n_frames, n_beams, n_samples)``, polar
    and un-scan-converted, so this is not an image. Alongside ``Tissue`` these
    files carry ``Ecg``, ``Resp`` and ``Aux2`` traces, and file-level attributes
    the MAT exports drop -- ``PatientID``, ``ManufacturerModelName``,
    ``ProbeName``, ``RecordingTime``, ``SWVersion``. (``ViewName`` is present
    but empty throughout this cohort, so the view still has to come from the
    filename.)

    Use it for that metadata or for raw-data work; :func:`read_clip` on
    ``<view>/data.mat`` remains the path to pixels.
    """
    import h5py

    with h5py.File(Path(path), "r") as handle:
        candidates = [
            key for key in handle.keys()
            if isinstance(handle[key], h5py.Dataset) and handle[key].ndim == 3
        ]
        if not candidates:
            raise ValueError(
                f"{Path(path).name}: no 3-D dataset; found {sorted(handle.keys())}"
            )
        named = [key for key in candidates if "tissue" in key.lower()]
        key = named[0] if named else candidates[0]
        attrs = {k: _decode_attr(v) for k, v in handle[key].attrs.items()}
        attrs.update({k: _decode_attr(v) for k, v in handle.attrs.items()})
        attrs["_dataset"] = key
        attrs["_candidates"] = candidates
        return np.asarray(handle[key]), attrs


def _decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    array = np.asarray(value)
    if array.size == 1:
        return array.ravel()[0]
    return array
