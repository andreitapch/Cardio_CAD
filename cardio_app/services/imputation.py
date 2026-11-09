from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold

from cardio_app.services.traces import matrix, ordered, split

SEGMENTS = 18
FOLDS = 5
MISSING_BANDS = {"1 or 2": (16, 17), "3 or 4": (14, 15), "5 or 6": (12, 13), "any": (1, 17)}
MASK_MODES = {"one segment per study": 1, "one wall per fold": 3, "one view per fold": 6}


@dataclass
class BestK:
    best_k: int
    lowest: float
    errors: pd.DataFrame
    rows: pd.DataFrame


@dataclass
class Imputed:
    rows: pd.DataFrame
    studies: list


def rrsse(actual, predicted):
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.sum((actual - predicted) ** 2)) / trapezoid(np.abs(actual)))


def masked(frame, mode, rng):
    frame = ordered(frame)
    if mode == 1:
        chosen = frame.groupby("study_nr").sample(n=1, random_state=rng).index
    else:
        column = "segment_wall" if mode == 3 else "cardiac_view"
        labels = frame[column].str.lower()
        chosen = frame.index[labels == rng.choice(np.sort(labels.unique()))]
    blanked = frame.copy()
    blanked["trace"] = blanked["trace"].map(np.array)
    blanked.loc[chosen, "trace"] = blanked.loc[chosen, "trace"].map(lambda t: np.full(t.size, np.nan))
    return blanked, frame.index.isin(chosen)


def best_k(frame, max_k, mode, seed=1):
    frame = ordered(frame)
    studies = frame["study_nr"].unique()
    rng = np.random.RandomState(seed)
    errors = np.zeros((max_k, FOLDS))
    collected = []
    for fold, (train_index, test_index) in enumerate(KFold(FOLDS).split(studies)):
        train = frame[frame["study_nr"].isin(studies[train_index])]
        test = frame[frame["study_nr"].isin(studies[test_index])].reset_index(drop=True)
        blanked, is_masked = masked(test, mode, rng)
        _, x_train, _ = matrix(train)
        test_studies, x_test, lengths = matrix(test)
        _, x_blank, _ = matrix(blanked)
        for k in range(1, max_k + 1):
            filled = KNNImputer(n_neighbors=k, weights="distance").fit(x_train).transform(x_blank)
            scores = []
            for study, row, blank, full in zip(test_studies, x_test, x_blank, filled):
                gap = np.isnan(blank)
                scores.append(rrsse(row[gap], full[gap]))
                for (position, original), imputed, flag in zip(
                        test.loc[test["study_nr"] == study, ["position", "trace"]].itertuples(index=False),
                        split(full, lengths[study]), is_masked[test["study_nr"] == study]):
                    if flag:
                        collected.append({"k": k, "fold": fold + 1, "study_nr": study, "position": position,
                                          "original": original, "imputed": imputed})
            errors[k - 1, fold] = np.mean(scores)
    means = errors.mean(axis=1)
    best = int(np.argmin(means)) + 1
    table = pd.DataFrame({"k": np.arange(1, max_k + 1), "mean RRSSE": means})
    rows = pd.DataFrame(collected)
    rows = rows[rows["k"] == best].reset_index(drop=True)
    labels = frame.drop_duplicates(["study_nr", "position"]).set_index(["study_nr", "position"])["curve_segment"]
    rows["label"] = [labels.get((s, p), str(p)) for s, p in zip(rows["study_nr"], rows["position"])]
    rows["rrsse"] = [rrsse(o, i) for o, i in zip(rows["original"], rows["imputed"])]
    return BestK(best, float(means[best - 1]), table, rows)


def with_missing_rows(frame, segments, source):
    frame = ordered(frame)
    length = int(frame["trace"].map(len).mode().iloc[0])
    config = segments[segments["segment_name"].str.strip().str.lower() != "apex"]
    added = []
    for study, group in frame.groupby("study_nr"):
        for row in config[~config["id_position"].isin(group["position"])].itertuples():
            view, wall = str(row.cardiac_view).strip(), str(row.cardiac_wall).strip().lower()
            if source == "Speqle":
                wall = str(row.seg_speqle).strip().lower()
            else:
                view = "aplax" if view.upper() == "3CH" else view.lower()
            added.append({"study_nr": study, "segment_wall": wall, "curve_length": length,
                          "curve_segment": str(row.segment_name).strip(), "id_position": str(row.id_position),
                          "cardiac_view": view, "trace": np.full(length, np.nan), "position": int(row.id_position),
                          "isImputed": 1})
    frame["isImputed"] = 0
    return ordered(pd.concat([frame, pd.DataFrame(added)], ignore_index=True))


def impute(complete, incomplete, k):
    _, x_train, _ = matrix(complete)
    studies, x_test, lengths = matrix(incomplete)
    filled = KNNImputer(n_neighbors=k, weights="distance").fit(x_train).transform(x_test)
    rows = ordered(incomplete).copy()
    pieces = [piece for study, vector in zip(studies, filled) for piece in split(vector, lengths[study])]
    rows["trace"] = [np.round(piece, 6) for piece in pieces]
    return Imputed(rows, list(studies))


def candidates(frame, band):
    low, high = band
    counts = frame.groupby("study_nr")["trace"].transform("size")
    return frame[(counts >= low) & (counts <= high)]
