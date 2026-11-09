"""On-disk cache of EchoCLIP embeddings, one ``.npy`` per clip.

Embeddings are a deterministic function of (clip, :class:`EncoderConfig`), so
they are computed once and reused. The GPU work is ~60 ms per clip; reading the
236 ``data.mat`` files it feeds on is 3.5 GB. Caching turns every subsequent
training run from minutes into milliseconds.

One file per clip rather than a single archive, because the build is the slow
part and it has to be resumable -- an interrupted pass picks up where it
stopped. Targets are deliberately *not* cached: they are cheap to read and
depend on a choice (which strain phase) that should stay changeable without
invalidating a GPU pass.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..loaders.dataset_index import ClipKey, ClipRef
from ..loaders.annotations.gt_strain import read_gt_strain
from ..loaders.images.image_prep import frames_first, prepare_clip
from ..loaders.annotations.scar_struct import read_view_analysis
from ..loaders.images.srd_reader import read_clip
from .config import DEFAULT_CACHE_DIR, EncoderConfig

__all__ = [
    "BuildReport",
    "EmbeddingDataset",
    "GROUPINGS",
    "clip_cache_path",
    "build_cache",
    "group_keys",
    "load_embeddings",
    "read_cache_meta",
]

META_FILENAME = "meta.json"

GROUPINGS = ("dataset_patient", "patient")


def group_keys(keys: list[ClipKey], by: str = "dataset_patient") -> np.ndarray:
    """Fold-grouping key per clip.

    ``dataset_patient`` -> ``"<dataset>/<patient>"``. Treats the two dataset
    folders as independent cohorts, so ``Retest_dataset/IV02`` and
    ``Test_dataset/IV02`` can fall in different folds.

    ``patient`` -> ``"<patient>"``. Keeps everything belonging to one person
    together, whichever folder it came from.

    Which is correct depends on whether the folders hold different people or
    the same people rescanned. The Vivid E9's own ``RecordingTime``, readable
    from the ``.h5`` exports, says the latter: of the 116 (patient, view) pairs
    present in both folders, 113 are the same PatientID on the same day minutes
    apart, 3 are byte-identical acquisitions, and none are on different days.
    On that evidence ``patient`` is the conservative choice and
    ``dataset_patient`` yields optimistic estimates by an unmeasured margin.
    The default follows the project's chosen protocol; use
    :meth:`EmbeddingDataset.regrouped` to quantify the gap.
    """
    if by == "dataset_patient":
        return np.asarray([f"{key.dataset}/{key.patient}" for key in keys])
    if by == "patient":
        return np.asarray([key.patient for key in keys])
    raise ValueError(f"grouping must be one of {list(GROUPINGS)}, got {by!r}")


@dataclass
class BuildReport:
    """What a cache build did, so the caller can report rather than guess."""

    written: list[ClipKey]
    reused: list[ClipKey]
    skipped: list[tuple[ClipKey, str]]
    failed: list[tuple[ClipKey, str]]

    @property
    def total(self) -> int:
        return len(self.written) + len(self.reused) + len(self.skipped) + len(self.failed)

    def summary(self) -> str:
        lines = [
            f"{self.total} clips: {len(self.written)} written, "
            f"{len(self.reused)} reused, {len(self.skipped)} skipped, "
            f"{len(self.failed)} failed"
        ]
        for key, reason in self.skipped:
            lines.append(f"  skip {key}: {reason}")
        for key, reason in self.failed:
            lines.append(f"  FAIL {key}: {reason}")
        return "\n".join(lines)


@dataclass
class EmbeddingDataset:
    """Everything a training run needs, assembled in memory.

    ``embeddings`` is ``(n_clips, n_frames, embed_dim)``; ``targets`` is
    ``(n_clips, 7)`` -- six segmental values then GLS, per
    :meth:`~motion_estimator.loaders.annotations.gt_strain.StrainGroundTruth.target_vector`.
    ``groups`` is the key cross-validation folds must be grouped on; see
    :func:`group_keys` for what it contains.

    23% of clips carry a NaN somewhere in ``targets`` -- GE could not track
    every wall, and it is segments 4-6 that suffer. :attr:`target_mask` marks
    what is real so a masked loss can use the rest of the clip instead of
    discarding it.
    """

    keys: list[ClipKey]
    embeddings: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    views: np.ndarray
    datasets: np.ndarray
    phase: str
    config: dict

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def target_mask(self) -> np.ndarray:
        """``(n_clips, 7)`` boolean -- True where the ground truth is real."""
        return np.isfinite(self.targets)

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.groups))

    @property
    def n_patients(self) -> int:
        """Distinct people, regardless of how ``groups`` is currently keyed."""
        return len({key.patient for key in self.keys})

    def support(self) -> np.ndarray:
        """How many clips have a real value for each of the 7 targets."""
        return self.target_mask.sum(axis=0)

    def regrouped(self, by: str) -> "EmbeddingDataset":
        """The same data with a different fold-grouping key. See :func:`group_keys`."""
        return replace(self, groups=group_keys(self.keys, by))

    def subset(self, mask) -> "EmbeddingDataset":
        """Rows where *mask* is true, e.g. one dataset folder."""
        mask = np.asarray(mask)
        return EmbeddingDataset(
            keys=[key for key, keep in zip(self.keys, mask) if keep],
            embeddings=self.embeddings[mask],
            targets=self.targets[mask],
            groups=self.groups[mask],
            views=self.views[mask],
            datasets=self.datasets[mask],
            phase=self.phase,
            config=self.config,
        )


def clip_cache_path(cache_dir: str | Path, key: ClipKey) -> Path:
    """Where one clip's embeddings live. Flat, so the directory is greppable."""
    return Path(cache_dir) / f"{key.dataset}__{key.patient}__{key.view}.npy"


def read_cache_meta(cache_dir: str | Path) -> dict | None:
    path = Path(cache_dir) / META_FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_cache(
    refs: list[ClipRef],
    encoder,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    progress=None,
) -> BuildReport:
    """Encode every complete clip in *refs* into *cache_dir*.

    Incomplete clips are skipped with a reason rather than dropped, and a clip
    that fails to read is recorded and the pass continues -- one bad file
    should not cost the other 235.

    A cache built with different :class:`EncoderConfig` settings is refused
    unless *overwrite* is set, since mixing fingerprints would silently produce
    a dataset whose rows are not comparable.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = encoder.config.fingerprint()
    existing = read_cache_meta(cache_dir)
    if existing and existing.get("config") != fingerprint and not overwrite:
        raise ValueError(
            f"{cache_dir} was built with different encoder settings:\n"
            f"  cached : {existing.get('config')}\n"
            f"  current: {fingerprint}\n"
            "Pass overwrite=True to rebuild, or point at a different directory."
        )

    report = BuildReport([], [], [], [])
    for ref in refs:
        if not ref.is_complete:
            report.skipped.append((ref.key, f"missing {ref.missing}"))
            continue

        destination = clip_cache_path(cache_dir, ref.key)
        if destination.is_file() and not overwrite:
            report.reused.append(ref.key)
            if progress:
                progress(ref.key, "reused")
            continue

        try:
            embeddings = encode_ref(ref, encoder)
        except Exception as error:
            report.failed.append((ref.key, f"{type(error).__name__}: {error}"))
            if progress:
                progress(ref.key, "failed")
            continue

        # Write via a temporary file so an interrupted pass cannot leave a
        # half-written .npy that a later run would happily reuse. Save through
        # an open handle: np.save appends ".npy" to any path that lacks it,
        # which would write "<name>.npy.tmp.npy" and leave the rename with
        # nothing to move.
        temporary = destination.with_name(destination.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.save(handle, embeddings.astype(np.float32))
        temporary.replace(destination)
        report.written.append(ref.key)
        if progress:
            progress(ref.key, "written")

    _write_meta(cache_dir, fingerprint, encoder)
    return report


def encode_ref(ref: ClipRef, encoder) -> np.ndarray:
    """One clip, from paths to ``(n_frames, embed_dim)``."""
    analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
    prepared = prepare_clip(
        read_clip(ref.clip_path), cycle_frame_limits=analysis.cycle_frame_limits
    )
    return encoder.encode_clip(frames_first(prepared.images))


def _write_meta(cache_dir: Path, fingerprint: dict, encoder) -> None:
    (cache_dir / META_FILENAME).write_text(
        json.dumps(
            {
                "config": fingerprint,
                "embed_dim": encoder.embed_dim,
                "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_embeddings(
    refs: list[ClipRef],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    phase: str = "end_systolic",
    grouping: str = "dataset_patient",
) -> EmbeddingDataset:
    """Assemble the cached embeddings and their targets into arrays.

    Only clips that are both complete and cached are included; anything else is
    a gap the caller should have seen in the build report, so it is left out
    here rather than silently zero-filled.
    """
    cache_dir = Path(cache_dir)
    meta = read_cache_meta(cache_dir)
    if meta is None:
        raise FileNotFoundError(
            f"no {META_FILENAME} in {cache_dir}; run build_cache first"
        )

    keys, rows, targets, views, datasets = [], [], [], [], []
    for ref in refs:
        path = clip_cache_path(cache_dir, ref.key)
        if not ref.is_complete or not path.is_file():
            continue
        ground_truth = read_gt_strain(ref.gt_strain_path)
        try:
            target = ground_truth.target_vector(ref.key.view, phase=phase)
        except KeyError:
            # The GT file has no entry for this view; not a cache problem.
            continue

        keys.append(ref.key)
        rows.append(np.load(path))
        targets.append(target)
        views.append(ref.key.view)
        datasets.append(ref.key.dataset)

    if not keys:
        raise ValueError(
            f"no cached clips matched the {len(refs)} refs given. Was the cache "
            f"built from the same index? ({cache_dir})"
        )

    shapes = {row.shape for row in rows}
    if len(shapes) != 1:
        raise ValueError(
            f"cached clips have inconsistent shapes {sorted(shapes)}; the cache "
            "was probably built across an encoder-settings change"
        )

    return EmbeddingDataset(
        keys=keys,
        embeddings=np.stack(rows),
        targets=np.asarray(targets, dtype=np.float64),
        groups=group_keys(keys, grouping),
        views=np.asarray(views),
        datasets=np.asarray(datasets),
        phase=phase,
        config=meta.get("config", {}),
    )
