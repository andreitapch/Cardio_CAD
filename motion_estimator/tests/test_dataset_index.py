"""Tests for cohort discovery, including the grouping rule that prevents leakage.

Run with:  python -m pytest motion_estimator/tests/test_dataset_index.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from motion_estimator.loaders.dataset_index import (
    ROOT_ENV_VAR,
    ClipKey,
    check_patient_id_collisions,
    dataset_root,
    index_dataset,
    manifest,
    summarize,
)


def test_finds_every_patient_and_view(fake_cohort: Path):
    refs = index_dataset(fake_cohort)

    assert len(refs) == 8  # 2 folders x 2 patients x 2 views
    assert {ref.key.view for ref in refs} == {"2ch", "4ch"}
    assert {ref.key.dataset for ref in refs} == {"Retest_dataset", "Test_dataset"}


def test_prefers_data_mat_over_srd(fake_cohort: Path):
    """``<view>/data.mat`` is what loadIntervendorData.m read, and always exists."""
    refs = index_dataset(fake_cohort)

    assert {ref.clip_source for ref in refs} == {"data.mat"}
    assert all(ref.clip_path.name == "data.mat" for ref in refs)


def test_does_not_silently_fall_back_to_srd(fake_cohort: Path):
    """The .srd is a coarser scan conversion, so substituting it mixes geometries.

    A missing data.mat has to read as a missing clip, not as a quietly
    different-resolution one.
    """
    (fake_cohort / "Retest_dataset" / "Data" / "IV02" / "2ch" / "data.mat").unlink()

    ref = _find(index_dataset(fake_cohort), "Retest_dataset", "IV02", "2ch")

    assert ref.clip_path is None
    assert ref.missing == ["clip"]
    assert ref.srd_path is not None  # it exists; it is deliberately not used


def test_any_opts_into_the_srd_fallback(fake_cohort: Path):
    (fake_cohort / "Retest_dataset" / "Data" / "IV02" / "2ch" / "data.mat").unlink()

    ref = _find(
        index_dataset(fake_cohort, clip_source="any"), "Retest_dataset", "IV02", "2ch"
    )

    assert ref.clip_source == "srd"
    assert ref.clip_path.name == "IV02_2ch.srd"


def test_any_still_prefers_data_mat(fake_cohort: Path):
    """data.mat wins when both exist -- it is the finer geometry."""
    refs = index_dataset(fake_cohort, clip_source="any")

    assert {ref.clip_source for ref in refs} == {"data.mat"}


def test_srd_only_reports_views_without_one_as_missing(fake_cohort: Path):
    """Retest ships no 4ch .srd at all, so an srd-only scan loses those clips."""
    refs = index_dataset(fake_cohort, clip_source="srd")

    two_ch = _find(refs, "Retest_dataset", "IV02", "2ch")
    four_ch = _find(refs, "Retest_dataset", "IV02", "4ch")

    assert two_ch.clip_source == "srd"
    assert four_ch.clip_path is None
    assert "clip" in four_ch.missing


def test_unknown_clip_source_is_rejected(fake_cohort: Path):
    with pytest.raises(ValueError, match="clip_source"):
        index_dataset(fake_cohort, clip_source="h5")


def test_clip_is_missing_when_no_source_exists(fake_cohort: Path):
    patient_dir = fake_cohort / "Retest_dataset" / "Data" / "IV02"
    (patient_dir / "4ch" / "data.mat").unlink()

    ref = _find(index_dataset(fake_cohort), "Retest_dataset", "IV02", "4ch")

    assert ref.clip_path is None
    assert ref.missing == ["clip"]
    assert not ref.is_complete


def test_view_discovered_from_an_h5_only_clip(fake_cohort: Path):
    """IV02 ships a .srd for 2ch but only an .h5 for 4ch in the real tree."""
    ref = _find(index_dataset(fake_cohort), "Retest_dataset", "IV02", "4ch")

    assert ref.h5_path is not None
    assert ref.srd_path is None


def test_missing_ground_truth_is_reported_not_dropped(fake_cohort: Path):
    """IV03 has no GT file; it must still appear, flagged."""
    refs = index_dataset(fake_cohort)
    iv03 = [ref for ref in refs if ref.key.patient == "IV03"]

    assert len(iv03) == 4
    assert all(ref.missing == ["gt_strain"] for ref in iv03)
    assert all(not ref.is_complete for ref in iv03)


def test_complete_clips_have_everything(fake_cohort: Path):
    complete = [ref for ref in index_dataset(fake_cohort) if ref.is_complete]

    assert len(complete) == 4
    assert all(ref.key.patient == "IV02" for ref in complete)
    assert all(ref.gt_strain_path is not None for ref in complete)
    assert all(ref.scar_struct_path is not None for ref in complete)


def test_group_id_is_scoped_by_dataset(fake_cohort: Path):
    """The whole point: Retest/IV02 and Test/IV02 are different people.

    Grouping folds on the bare patient id would put two unrelated patients in
    one group and leak across the train/test boundary.
    """
    refs = index_dataset(fake_cohort)
    groups = {ref.group_id for ref in refs}

    assert groups == {
        "Retest_dataset/IV02",
        "Retest_dataset/IV03",
        "Test_dataset/IV02",
        "Test_dataset/IV03",
    }
    assert len({ref.key.patient for ref in refs}) == 2  # ids alone collapse them


def test_both_views_of_a_patient_share_a_group(fake_cohort: Path):
    """A patient's 2ch and 4ch clips must never straddle a fold boundary."""
    refs = index_dataset(fake_cohort)
    two_ch = _find(refs, "Retest_dataset", "IV02", "2ch")
    four_ch = _find(refs, "Retest_dataset", "IV02", "4ch")

    assert two_ch.group_id == four_ch.group_id


def test_collisions_are_surfaced(fake_cohort: Path):
    collisions = check_patient_id_collisions(index_dataset(fake_cohort))

    assert collisions == {
        "IV02": ["Retest_dataset", "Test_dataset"],
        "IV03": ["Retest_dataset", "Test_dataset"],
    }


def test_can_index_a_single_folder(fake_cohort: Path):
    refs = index_dataset(fake_cohort, datasets=("Retest_dataset",))

    assert {ref.key.dataset for ref in refs} == {"Retest_dataset"}
    assert not check_patient_id_collisions(refs)


def test_unknown_vendor_leaves_gt_missing(fake_cohort: Path):
    refs = index_dataset(fake_cohort, vendor="Philips")

    assert all(ref.gt_strain_path is None for ref in refs)


def test_non_patient_directories_are_ignored(fake_cohort: Path):
    (fake_cohort / "Retest_dataset" / "Data" / "notes").mkdir()

    assert len(index_dataset(fake_cohort)) == 8


def test_absent_root_names_the_env_var(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match=ROOT_ENV_VAR):
        index_dataset(tmp_path / "nope")


def test_dataset_root_precedence(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ROOT_ENV_VAR, str(tmp_path / "from_env"))

    assert dataset_root() == tmp_path / "from_env"
    assert dataset_root(tmp_path / "explicit") == tmp_path / "explicit"


def test_manifest_has_a_row_per_clip(fake_cohort: Path):
    pytest.importorskip("pandas")
    frame = manifest(index_dataset(fake_cohort))

    assert len(frame) == 8
    assert frame["complete"].sum() == 4
    assert set(frame["group_id"]) == {
        "Retest_dataset/IV02",
        "Retest_dataset/IV03",
        "Test_dataset/IV02",
        "Test_dataset/IV03",
    }


def test_summary_reports_counts_and_gaps(fake_cohort: Path):
    text = summarize(index_dataset(fake_cohort))

    assert "8 clips discovered" in text
    assert "Retest_dataset: 2 patients, 4 clips (2 complete)" in text
    assert "missing ['gt_strain']" in text
    assert "group on group_id" in text


def test_clip_key_str():
    assert str(ClipKey("Retest_dataset", "IV02", "2ch")) == "Retest_dataset/IV02/2ch"


def _find(refs, dataset, patient, view):
    key = ClipKey(dataset, patient, view)
    matches = [ref for ref in refs if ref.key == key]
    assert len(matches) == 1, f"{key} not found among {len(refs)} refs"
    return matches[0]


def test_only_default_views_are_indexed(fake_cohort: Path):
    """Test_dataset ships 3ch clips that scar_struct.mat defines no window for."""
    patient_dir = fake_cohort / "Test_dataset" / "Data" / "IV02"
    (patient_dir / "3ch").mkdir()
    (patient_dir / "IV02_3ch.srd").write_bytes(b"")

    refs = index_dataset(fake_cohort)

    assert "3ch" not in {ref.key.view for ref in refs}
    assert len(refs) == 8


def test_views_none_indexes_everything_found(fake_cohort: Path):
    patient_dir = fake_cohort / "Test_dataset" / "Data" / "IV02"
    (patient_dir / "3ch").mkdir()

    refs = index_dataset(fake_cohort, views=None)

    assert "3ch" in {ref.key.view for ref in refs}
    assert len(refs) == 9


def test_views_can_be_narrowed_further(fake_cohort: Path):
    refs = index_dataset(fake_cohort, views=("4ch",))

    assert {ref.key.view for ref in refs} == {"4ch"}
    assert len(refs) == 4


def test_verify_views_is_off_by_default(fake_cohort: Path):
    """A plain scan touches no MAT files, so the flag stays unknown."""
    refs = index_dataset(fake_cohort)

    assert all(ref.scar_struct_has_view is None for ref in refs)


def test_verify_views_flags_a_view_scar_struct_lacks(fake_cohort: Path):
    """This is the 3ch failure, caught at index time instead of at load time.

    The 3ch clip itself is present and readable -- what it lacks is an
    ``info_3ch`` entry, and therefore a cardiac-cycle window.
    """
    import numpy as np

    from .conftest import make_clip_mat

    patient_dir = fake_cohort / "Test_dataset" / "Data" / "IV02"
    (patient_dir / "3ch").mkdir()
    make_clip_mat(
        patient_dir / "3ch" / "data.mat", np.zeros((4, 3, 100), dtype=np.uint8)
    )

    refs = index_dataset(fake_cohort, views=None, verify_views=True)
    three_ch = _find(refs, "Test_dataset", "IV02", "3ch")
    two_ch = _find(refs, "Test_dataset", "IV02", "2ch")

    assert three_ch.scar_struct_has_view is False
    assert three_ch.missing == ["scar_struct_view"]
    assert not three_ch.is_complete
    assert two_ch.scar_struct_has_view is True
    assert two_ch.is_complete


def test_verify_views_confirms_the_default_views(fake_cohort: Path):
    refs = index_dataset(fake_cohort, verify_views=True)

    assert all(ref.scar_struct_has_view is True for ref in refs)


def test_unreadable_scar_struct_marks_the_view_missing(fake_cohort: Path):
    """A corrupt scar_struct must not abort the whole scan."""
    scar = fake_cohort / "Retest_dataset" / "Data" / "IV02" / "Analysis" / "scar_struct.mat"
    scar.write_bytes(b"not a mat file")

    ref = _find(index_dataset(fake_cohort, verify_views=True), "Retest_dataset", "IV02", "2ch")

    assert ref.scar_struct_has_view is False
    assert "scar_struct_view" in ref.missing
