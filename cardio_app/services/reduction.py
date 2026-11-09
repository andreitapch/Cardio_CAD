import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cardio_app.services import autoencoder, pca, traces

MODES = {"full traces": "full_traces", "per segment": "segment"}
METHODS = {"PCA": "PCA", "Autoencoder": "AutoencoderDm"}
FULL = "full"


@dataclass
class Block:
    name: str
    studies: np.ndarray
    x: np.ndarray
    lengths: list
    positions: list
    segments: list
    n_trace: int
    scaler: StandardScaler | None = None


@dataclass
class Result:
    method: str
    codes: dict = field(default_factory=dict)
    recon: dict = field(default_factory=dict)
    k: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)


def blocks(frame, mode, echo=None):
    frame = traces.ordered(frame)
    groups = {FULL: frame} if mode == "full_traces" else dict(tuple(frame.groupby("curve_segment")))
    out = []
    for name, group in groups.items():
        studies, x, lengths = traces.matrix(group)
        by_study = traces.ordered(group).groupby("study_nr", sort=False)
        block = Block(name, studies, x, lengths.tolist(), by_study["position"].apply(list).tolist(),
                      by_study["curve_segment"].apply(list).tolist(), x.shape[1])
        if echo is not None:
            block.scaler = StandardScaler().fit(x)
            block.x = np.hstack([block.scaler.transform(x), echo.reindex(studies).fillna(0.0).to_numpy(float)])
        out.append(block)
    return out


def save_blocks(path, items):
    np.savez(path, names=np.array([b.name for b in items]), **{f"x__{b.name}": b.x for b in items})


def load_blocks(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[f"x__{name}"] for name in data["names"]}


def baseline(x):
    mean = x.mean(axis=0)
    return {"method": "Mean curve", "k": 0, "n": len(x), "features": x.shape[1], "mse": float(np.mean((x - mean) ** 2)),
            "r2": 0.0, "train_r2": 0.0, "holdout_r2": 0.0, "cum_var": None, "params": int(mean.size)}


def reduce_pca(x, request):
    model = pca.fit(x, request)
    codes = model.transform(x)
    recon = model.inverse_transform(codes)
    split = autoencoder.split_indices(len(x))
    train_r2, holdout_r2 = pca.holdout(x, model.n_components_, *split) if split else (float("nan"), float("nan"))
    metrics = {"method": "PCA", "k": int(model.n_components_), "n": len(x), "features": x.shape[1],
               "mse": float(np.mean((x - recon) ** 2)), "r2": pca.r2(model, x), "train_r2": train_r2,
               "holdout_r2": holdout_r2, "cum_var": float(np.sum(model.explained_variance_ratio_)),
               "params": int(model.components_.size + model.mean_.size)}
    params = {"components": model.components_, "mean": model.mean_, "explained_variance": model.explained_variance_,
              "explained_variance_ratio": model.explained_variance_ratio_, "singular_values": model.singular_values_}
    return codes, recon, metrics, params


def reduce_autoencoder(x, k, epochs, progress=None):
    model, history = autoencoder.fit(x, k, epochs, progress=progress)
    codes = autoencoder.encode(model, x)
    recon = autoencoder.decode(model, codes)
    data = autoencoder.tensor(x)
    split = autoencoder.split_indices(len(x))
    train_r2 = autoencoder.r2(model, data[split[1]]) if split else float("nan")
    holdout_r2 = autoencoder.r2(model, data[split[0]]) if split else float("nan")
    metrics = {"method": "Autoencoder", "k": int(k), "n": len(x), "features": x.shape[1],
               "mse": autoencoder.mse(model, data), "r2": autoencoder.r2(model, data), "train_r2": train_r2,
               "holdout_r2": holdout_r2, "cum_var": None, "params": int(sum(p.numel() for p in model.parameters()))}
    return codes, recon, metrics, history


def save_result(path, results):
    arrays, metrics = {}, {}
    for result in results:
        metrics[result.method] = {"metrics": result.metrics, "k": result.k}
        for name in result.codes:
            arrays[f"{result.method}__codes__{name}"] = result.codes[name]
            arrays[f"{result.method}__recon__{name}"] = result.recon[name]
            if name in result.history:
                arrays[f"{result.method}__history__{name}"] = result.history[name]
            for key, value in result.models.get(name, {}).items():
                arrays[f"{result.method}__model__{name}__{key}"] = value
    np.savez(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metrics), encoding="utf-8")


def load_result(path):
    summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    results = {method: Result(method, metrics=entry["metrics"], k=entry["k"]) for method, entry in summary.items()}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            method, kind, name, *rest = key.split("__")
            result = results[method]
            if kind == "model":
                result.models.setdefault(name, {})[rest[0]] = data[key]
            else:
                getattr(result, kind)[name] = data[key]
    return results


def reconstructed_rows(block, recon):
    recon = recon[:, :block.n_trace]
    if block.scaler is not None:
        recon = block.scaler.inverse_transform(recon)
    rows = []
    for study, vector, lengths, positions, segments in zip(block.studies, recon, block.lengths, block.positions, block.segments):
        for piece, position, segment in zip(traces.split(vector, lengths), positions, segments):
            rows.append({"study_nr": study, "curve_segment": segment, "id_position": position, "trace": np.round(piece, 6)})
    return pd.DataFrame(rows)


def reconstructed_frame(items, result):
    return traces.ordered(pd.concat([reconstructed_rows(block, result.recon[block.name]) for block in items], ignore_index=True))


def stitched_codes(items, result):
    studies = sorted(set().union(*[set(block.studies) for block in items]))
    parts = []
    for block in sorted(items, key=lambda b: b.name):
        codes = result.codes[block.name]
        index = {study: i for i, study in enumerate(block.studies)}
        part = np.full((len(studies), codes.shape[1]), np.nan)
        for row, study in enumerate(studies):
            if study in index:
                part[row] = codes[index[study]]
        parts.append(part)
    return np.array(studies), np.hstack(parts)


def summary_table(result, baseline_metrics=None):
    rows = [baseline_metrics] if baseline_metrics else []
    rows += [dict(m, block=name) for name, m in result.metrics.items()]
    return pd.DataFrame(rows)


def mean_metrics(result, label):
    frame = pd.DataFrame(list(result.metrics.values()))
    row = {"method": label, "k": int(frame["k"].iloc[0]), "n": int(frame["n"].iloc[0]), "features": int(frame["features"].iloc[0]),
           "params": int(frame["params"].sum())}
    for key in ("mse", "r2", "train_r2", "holdout_r2", "cum_var"):
        row[key] = float(frame[key].dropna().mean()) if frame[key].notna().any() else None
    return row


def study_mse(original, reconstructed, study):
    left = original[original["study_nr"] == study].set_index("position")["trace"]
    right = reconstructed[reconstructed["study_nr"] == study].set_index("position")["trace"]
    return float(np.mean([np.mean((left[p] - right[p]) ** 2) for p in left.index if p in right.index]))
