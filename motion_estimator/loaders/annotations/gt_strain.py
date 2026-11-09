"""Ground-truth strain from ``GT_STRAIN/<vendor>_mat/<prefix><patient>.mat``.

Each file holds a single ``strain`` struct with, per cardiac view, three
six-element segmental vectors and one end-systolic global value::

    segm_2ch_peak_systolic_strain   1x6
    segm_2ch_end_systolic_strain    1x6
    segm_2ch_post_systolic_strain   1x6
    segm_4ch_...                    (the same three)
    GLS_2ch_ES, GLS_4ch_ES          scalars

The distinction between the phases is the one ``mainComputeStrain.m`` draws:
``valES = strainCurve(ED)`` samples the curve at end-systole, while
``valPeakS = min(strainCurve(1:ED))`` takes its extreme. They coincide for
segments that have not begun to recover -- IV02's 2ch vectors differ only in
segment 4 -- so mixing the two up is easy and mostly invisible.

The training target is the **end-systolic** set, which is the phase
``GLS_*_ES`` is defined at; all three are loaded so that choice stays visible
at the point of use rather than baked in here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..images import _matio

__all__ = [
    "StrainGroundTruth",
    "PHASES",
    "N_SEGMENTS",
    "vendor_location",
    "gt_path",
    "read_gt_strain",
]

N_SEGMENTS = 6
PHASES = ("peak_systolic", "end_systolic", "post_systolic")

# Folder and file prefix per vendor, as wired up in loadOptions.m:58-62. The
# regular pattern is "<Vendor>_mat" / "<Vendor>_strain"; these two break it.
_VENDOR_OVERRIDES = {
    "Average": ("Average_mat", "Mean_vendors_strain"),
    "Not_GE_Mean": ("Not_GE_Mean", "NOT_GE_Mean"),
}

_SEGMENTAL_FIELD = re.compile(
    r"^segm_(?P<view>\w+?)_(?P<phase>peak_systolic|end_systolic|post_systolic)_strain$"
)
_GLS_FIELD = re.compile(r"^GLS_(?P<view>\w+?)_ES$")


@dataclass
class StrainGroundTruth:
    """One patient's strain ground truth from one vendor.

    ``segmental[phase][view]`` is a ``(6,)`` array in percent, negative for
    shortening. ``gls_es[view]`` is that view's end-systolic global
    longitudinal strain.
    """

    patient: str
    vendor: str
    segmental: dict[str, dict[str, np.ndarray]]
    gls_es: dict[str, float]
    source: Path | None = None

    @property
    def views(self) -> list[str]:
        return sorted(self.gls_es)

    def target_vector(self, view: str, phase: str = "end_systolic") -> np.ndarray:
        """The 7-element regression target: 6 segmental values then GLS.

        This is the layout the model head predicts, so keep it as the single
        definition of that ordering.
        """
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        try:
            segments = self.segmental[phase][view]
        except KeyError:
            raise KeyError(
                f"{self.patient}/{self.vendor}: no {phase} strain for view "
                f"{view!r}; available views: {self.views}"
            ) from None
        if view not in self.gls_es:
            raise KeyError(
                f"{self.patient}/{self.vendor}: no GLS for view {view!r}"
            )
        return np.concatenate([segments, [self.gls_es[view]]])


def vendor_location(vendor: str) -> tuple[str, str]:
    """``(folder, file_prefix)`` for *vendor* under ``GT_STRAIN``."""
    if vendor in _VENDOR_OVERRIDES:
        return _VENDOR_OVERRIDES[vendor]
    return f"{vendor}_mat", f"{vendor}_strain"


def gt_path(dataset_dir: str | Path, patient: str, vendor: str = "GE") -> Path:
    """Where *patient*'s *vendor* ground truth lives under a dataset folder.

    *dataset_dir* is a ``Retest_dataset`` / ``Test_dataset`` directory.
    """
    folder, prefix = vendor_location(vendor)
    return Path(dataset_dir) / "GT_STRAIN" / folder / f"{prefix}{patient}.mat"


def read_gt_strain(path: str | Path, patient: str = "", vendor: str = "") -> StrainGroundTruth:
    """Parse one ``GT_STRAIN`` file.

    *patient* and *vendor* are labels only; when omitted they are recovered
    from the filename and its parent folder.
    """
    path = Path(path)
    variables = _matio.load_mat(path)
    if "strain" not in variables:
        raise ValueError(
            f"{path.name}: no 'strain' variable; found {sorted(variables)}"
        )
    strain = variables["strain"]

    segmental: dict[str, dict[str, np.ndarray]] = {phase: {} for phase in PHASES}
    gls_es: dict[str, float] = {}

    for name in _matio.fields(strain):
        segmental_match = _SEGMENTAL_FIELD.match(name)
        if segmental_match:
            view = segmental_match.group("view")
            phase = segmental_match.group("phase")
            values = np.asarray(_matio.field(strain, name), dtype=float).ravel()
            if values.size != N_SEGMENTS:
                raise ValueError(
                    f"{path.name}: {name} has {values.size} values, "
                    f"expected {N_SEGMENTS}"
                )
            segmental[phase][view] = values
            continue

        gls_match = _GLS_FIELD.match(name)
        if gls_match:
            gls_es[gls_match.group("view")] = float(
                _matio.scalar(_matio.field(strain, name))
            )

    if not gls_es:
        raise ValueError(
            f"{path.name}: no GLS_<view>_ES fields; found "
            f"{list(_matio.fields(strain))}"
        )

    return StrainGroundTruth(
        patient=patient or _patient_from_name(path),
        vendor=vendor or path.parent.name.replace("_mat", ""),
        segmental=segmental,
        gls_es=gls_es,
        source=path,
    )


def _patient_from_name(path: Path) -> str:
    """``GE_strainIV02.mat`` -> ``IV02``."""
    match = re.search(r"(IV\d+)", path.stem)
    return match.group(1) if match else path.stem
