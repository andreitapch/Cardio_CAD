"""Cohort discovery over the 2D GE intervendor in-vivo tree.

Expected layout, one level per dataset folder::

    <root>/Retest_dataset/Data/IV02/IV02_2ch.srd        # not always present
                                   /IV02_2ch.h5
                                   /Analysis/scar_struct.mat
                                   /2ch/data.mat        # the canonical clip
                                   /4ch/data.mat
           /Retest_dataset/GT_STRAIN/GE_mat/GE_strainIV02.mat
    <root>/Test_dataset/...

Two things this module exists to get right:

**The clip source is ``<view>/data.mat``, and only that.** The three exports in
a patient directory are one acquisition at three different geometries -- see
:mod:`motion_estimator.loaders.images.srd_reader` -- and the ``.srd`` is a *coarser* scan
conversion (0.75 mm, ~250x200) than ``data.mat`` (~0.3 mm, 500x500). Falling
back from one to the other would hand the model two different resolutions
depending on which files a patient happened to ship, so
:func:`index_dataset` defaults to ``data.mat`` alone and reports a clip as
missing rather than substituting. It costs nothing on this cohort: all 236
2ch/4ch clips have a ``data.mat``, while Retest ships no 4ch ``.srd`` at all.
``data.mat`` is also what ``loadIntervendorData.m:15`` read.

**The group key.** ``Retest_dataset`` and ``Test_dataset`` are independent
cohorts that reuse the same ``IV##`` numbering, so a patient identifier alone
is *not* an identity: ``Retest_dataset/IV02`` and ``Test_dataset/IV02`` are
different people. Grouping cross-validation folds on the bare patient id would
fuse them and leak across the split. :attr:`ClipRef.group_id` is the composite
key, and :func:`check_patient_id_collisions` makes the overlap explicit rather
than letting it pass unnoticed.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .annotations.gt_strain import gt_path

__all__ = [
    "DEFAULT_ROOT",
    "DATASET_FOLDERS",
    "DEFAULT_VIEWS",
    "CLIP_SOURCES",
    "ClipKey",
    "ClipRef",
    "dataset_root",
    "index_dataset",
    "manifest",
    "summarize",
    "check_patient_id_collisions",
]

# Overridable because the dataset lives on an external drive whose letter moves.
DEFAULT_ROOT = Path(r"F:\PHD\DataSets\2D_GE_intervendor_in_vivo")
ROOT_ENV_VAR = "MOTION_ESTIMATOR_DATA_ROOT"

# loadOptions.m:55 -- both folders, in this order.
DATASET_FOLDERS = ("Retest_dataset", "Test_dataset")

DATA_MAT = "data.mat"
SRD = "srd"
ANY_CLIP = "any"
CLIP_SOURCES = (DATA_MAT, SRD, ANY_CLIP)

# Test_dataset also ships 3ch clips, but scar_struct.mat only ever defines
# info_2ch and info_4ch, so a 3ch clip has no cardiac-cycle window and cannot
# be cropped or matched to the GE ground truth (which is likewise 2ch/4ch
# only). Restricting the scan is the fix; pass views=None to see everything.
DEFAULT_VIEWS = ("2ch", "4ch")

_PATIENT_DIR = re.compile(r"^IV\d+$")
_VIEW_DIR = re.compile(r"^\d+ch$")


@dataclass(frozen=True)
class ClipKey:
    """Identifies one clip: a dataset folder, a patient, and a cardiac view."""

    dataset: str
    patient: str
    view: str

    def __str__(self) -> str:
        return f"{self.dataset}/{self.patient}/{self.view}"


@dataclass
class ClipRef:
    """Resolved paths for one clip, with whatever is absent recorded."""

    key: ClipKey
    clip_path: Path | None = None
    clip_source: str | None = None
    data_mat_path: Path | None = None
    srd_path: Path | None = None
    h5_path: Path | None = None
    scar_struct_path: Path | None = None
    gt_strain_path: Path | None = None
    scar_struct_has_view: bool | None = None

    @property
    def group_id(self) -> str:
        """Cross-validation group. Scoped by dataset -- see the module docstring."""
        return f"{self.key.dataset}/{self.key.patient}"

    @property
    def missing(self) -> list[str]:
        """Names of the pieces needed for training that this clip lacks."""
        absent = []
        if self.clip_path is None:
            absent.append("clip")
        if self.scar_struct_path is None:
            absent.append("scar_struct")
        elif self.scar_struct_has_view is False:
            # The file exists but defines no info_<view> for this view, so
            # there is no cardiac-cycle window to crop to. Only populated when
            # index_dataset ran with verify_views=True.
            absent.append("scar_struct_view")
        if self.gt_strain_path is None:
            absent.append("gt_strain")
        return absent

    @property
    def is_complete(self) -> bool:
        """Whether this clip can contribute a training example."""
        return not self.missing


def dataset_root(root: str | Path | None = None) -> Path:
    """Resolve the dataset root: explicit argument, then env var, then default."""
    if root is not None:
        return Path(root)
    from_env = os.environ.get(ROOT_ENV_VAR)
    return Path(from_env) if from_env else DEFAULT_ROOT


def index_dataset(
    root: str | Path | None = None,
    datasets: tuple[str, ...] = DATASET_FOLDERS,
    vendor: str = "GE",
    views: tuple[str, ...] | None = DEFAULT_VIEWS,
    verify_views: bool = False,
    clip_source: str = DATA_MAT,
) -> list[ClipRef]:
    """Walk *root* and return one :class:`ClipRef` per discovered clip.

    Incomplete clips are returned too, with :attr:`ClipRef.missing` populated,
    so the caller reports what is absent instead of the scan hiding it.

    *views* restricts which cardiac views are indexed, defaulting to
    :data:`DEFAULT_VIEWS`; pass ``None`` to index every view found on disk.

    *verify_views* additionally opens each ``scar_struct.mat`` to confirm it
    actually defines the view, marking the clip ``scar_struct_view``-missing if
    not. Off by default so a plain scan stays pure filesystem metadata; the
    cost is one small read per patient.

    *clip_source* picks which export to read: ``"data.mat"`` (the default, and
    the only one with a uniform geometry across the cohort), ``"srd"``, or
    ``"any"`` to prefer ``data.mat`` and fall back to the ``.srd``. Prefer the
    default -- see the module docstring for why mixing them is a hazard rather
    than a convenience.
    """
    if clip_source not in CLIP_SOURCES:
        raise ValueError(
            f"clip_source must be one of {list(CLIP_SOURCES)}, got {clip_source!r}"
        )
    root = dataset_root(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"dataset root {root} is not a directory. Set the "
            f"{ROOT_ENV_VAR} environment variable if the drive letter changed."
        )

    refs: list[ClipRef] = []
    seen: set[ClipKey] = set()

    for dataset in datasets:
        data_dir = root / dataset / "Data"
        if not data_dir.is_dir():
            continue
        dataset_dir = root / dataset

        for patient_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            if not _PATIENT_DIR.match(patient_dir.name):
                continue
            patient = patient_dir.name

            scar = patient_dir / "Analysis" / "scar_struct.mat"
            scar = scar if scar.is_file() else None
            gt = gt_path(dataset_dir, patient, vendor)
            gt = gt if gt.is_file() else None

            scar_views = None
            if verify_views and scar is not None:
                scar_views = set(_scar_struct_views(scar))

            for view in _discover_views(patient_dir, patient, views):
                key = ClipKey(dataset, patient, view)
                if key in seen:
                    raise RuntimeError(f"duplicate clip key {key}")
                seen.add(key)
                refs.append(
                    _build_ref(
                        key,
                        patient_dir,
                        patient,
                        view,
                        scar,
                        gt,
                        scar_views,
                        clip_source,
                    )
                )

    return refs


def _discover_views(
    patient_dir: Path, patient: str, allowed: tuple[str, ...] | None
) -> list[str]:
    """View names for a patient, from both the view folders and the clip files."""
    found = {
        entry.name
        for entry in patient_dir.iterdir()
        if entry.is_dir() and _VIEW_DIR.match(entry.name)
    }
    for suffix in (".srd", ".h5"):
        for clip in patient_dir.glob(f"{patient}_*{suffix}"):
            view = clip.stem[len(patient) + 1:]
            if _VIEW_DIR.match(view):
                found.add(view)
    if allowed is not None:
        found &= set(allowed)
    return sorted(found)


def _scar_struct_views(scar: Path) -> list[str]:
    """Views defined in a ``scar_struct.mat``, or none if it will not open."""
    from .annotations.scar_struct import available_views

    try:
        return available_views(scar)
    except Exception:
        return []


def _build_ref(
    key: ClipKey,
    patient_dir: Path,
    patient: str,
    view: str,
    scar: Path | None,
    gt: Path | None,
    scar_views: set[str] | None = None,
    clip_source: str = DATA_MAT,
) -> ClipRef:
    data_mat = patient_dir / view / "data.mat"
    data_mat = data_mat if data_mat.is_file() else None
    srd = patient_dir / f"{patient}_{view}.srd"
    srd = srd if srd.is_file() else None
    h5 = patient_dir / f"{patient}_{view}.h5"
    h5 = h5 if h5.is_file() else None

    if clip_source == DATA_MAT:
        candidates = ((data_mat, DATA_MAT),)
    elif clip_source == SRD:
        candidates = ((srd, SRD),)
    else:  # ANY_CLIP -- data.mat first, since its geometry is the finer one
        candidates = ((data_mat, DATA_MAT), (srd, SRD))

    resolved_path, resolved_source = None, None
    for path, label in candidates:
        if path is not None:
            resolved_path, resolved_source = path, label
            break

    return ClipRef(
        key=key,
        clip_path=resolved_path,
        clip_source=resolved_source,
        data_mat_path=data_mat,
        srd_path=srd,
        h5_path=h5,
        scar_struct_path=scar,
        gt_strain_path=gt,
        scar_struct_has_view=None if scar_views is None else view in scar_views,
    )


def check_patient_id_collisions(refs: list[ClipRef]) -> dict[str, list[str]]:
    """Patient ids that appear in more than one dataset folder.

    Expected to be non-empty for this cohort, since the two folders number
    their patients independently. Returned so a caller can state the overlap
    plainly; it is only a problem if something groups on the bare id.
    """
    by_patient: dict[str, set[str]] = {}
    for ref in refs:
        by_patient.setdefault(ref.key.patient, set()).add(ref.key.dataset)
    return {
        patient: sorted(datasets)
        for patient, datasets in sorted(by_patient.items())
        if len(datasets) > 1
    }


def manifest(refs: list[ClipRef]):
    """One row per clip as a ``pandas.DataFrame``, for eyeballing the cohort."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "dataset": ref.key.dataset,
                "patient": ref.key.patient,
                "view": ref.key.view,
                "group_id": ref.group_id,
                "clip_source": ref.clip_source,
                "has_srd": ref.srd_path is not None,
                "has_h5": ref.h5_path is not None,
                "has_scar_struct": ref.scar_struct_path is not None,
                "scar_struct_has_view": ref.scar_struct_has_view,
                "has_gt_strain": ref.gt_strain_path is not None,
                "complete": ref.is_complete,
                "missing": ",".join(ref.missing),
            }
            for ref in refs
        ]
    )


def summarize(refs: list[ClipRef]) -> str:
    """Human-readable cohort summary: counts per folder, views, and gaps."""
    lines = [f"{len(refs)} clips discovered"]

    for dataset in sorted({ref.key.dataset for ref in refs}):
        subset = [ref for ref in refs if ref.key.dataset == dataset]
        patients = sorted({ref.key.patient for ref in subset})
        complete = [ref for ref in subset if ref.is_complete]
        views = sorted({ref.key.view for ref in subset})
        lines.append(
            f"  {dataset}: {len(patients)} patients, {len(subset)} clips "
            f"({len(complete)} complete), views {views}"
        )
        for ref in subset:
            if ref.missing:
                lines.append(f"      {ref.key} missing {ref.missing}")

    collisions = check_patient_id_collisions(refs)
    if collisions:
        lines.append(
            f"  {len(collisions)} patient id(s) appear in both folders "
            "(different people; group on group_id, not patient)"
        )
    return "\n".join(lines)
