from functools import lru_cache

import numpy as np
import pandas as pd

from motion_estimator.viewer import catalog

MITT = catalog.MITT


def models():
    return {model.id: model.label for model in catalog.models().values()}


def clip_keys():
    return catalog.clip_keys()


@lru_cache(maxsize=4)
def frames(key):
    with np.load(catalog.clip_file(key), allow_pickle=False) as data:
        return data["frames"]


def _mtime(path):
    return path.stat().st_mtime if path.is_file() else 0.0


@lru_cache(maxsize=2048)
def _result(model_id, key, stamp):
    if not stamp:
        return None
    with np.load(catalog.result_file(model_id, key), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def result(model_id, key):
    if not model_id:
        return None
    return _result(model_id, key, _mtime(catalog.result_file(model_id, key)))


@lru_cache(maxsize=4)
def _flow(model_id, key, stamp):
    return np.load(catalog.flow_file(model_id, key)).astype(np.float32)


def flow(model_id, key):
    path = catalog.flow_file(model_id, key)
    return _flow(model_id, key, _mtime(path)) if path.is_file() else None


def table(model_a, model_b, dataset, view):
    rows = []
    for key in clip_keys():
        ds, patient, v = key.split("/")
        if (dataset != "all" and ds != dataset) or (view != "all" and v != view):
            continue
        row = {"clip": key, "dataset": ds.split("_")[0], "patient": patient, "view": v}
        for slot, model in (("a", model_a), ("b", model_b)):
            r = result(model, key)
            row[f"epe_{slot}"] = round(float(r["epe_mm"]), 3) if r is not None and np.isfinite(r["epe_mm"]) else np.nan
            row[f"seen_{slot}"] = "trained on" if r is not None and bool(r["seen"]) else ""
        row["diff"] = round(row["epe_b"] - row["epe_a"], 3)
        rows.append(row)
    return pd.DataFrame(rows)
