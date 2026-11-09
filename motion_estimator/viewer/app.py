"""Cardiac motion viewer, part 1: pick a patient, compare two models with MITT, play the sequence.

    python -m motion_estimator.viewer.app            # then open http://127.0.0.1:8050

* **Patients**: the 236 in vivo clips (Retest + Test, 2ch + 4ch), with each
  model's contour EPE against MITT (the paper's metric, mm) -- sortable, so the
  worst cases are one click away.
* **Models A and B**: any trained run. Cross-validated tests use, for each
  patient, the fold that never trained on it. A run that trained on the
  patient is flagged.
* **Sequence**: echo frames with MITT's contour (white, dashed) and the two
  models' contours (blue A, orange B) propagated from MITT's frame-1 contour,
  optionally the colour-coded flow of A or B; per-frame EPE; longitudinal
  strain with GE's GLS and the ES frame.
* **Predict** runs the GPU worker (:mod:`.worker`) in its own process for the
  selected clip (with flow), or for all clips; results are cached, so this is
  needed once per model.

New recordings (DICOM, .srd, .h5) come in part 2; they need a starting contour.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dash_table, dcc, html
from matplotlib.colors import hsv_to_rgb
from PIL import Image

from . import catalog

REPO = Path(__file__).resolve().parents[2]
JOBS = catalog.VIEWER_CACHE / "_jobs"
A_COLOUR, B_COLOUR, MITT_COLOUR = "#3987e5", "#eb6834", "#ffffff"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# -- data -------------------------------------------------------------------

def _png(array: np.ndarray, mode: str) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@lru_cache(maxsize=4)
def clip_frames(key: str) -> tuple[list[str], int, int]:
    """Every frame as a PNG data URI, plus height and width."""
    with np.load(catalog.clip_file(key), allow_pickle=False) as data:
        frames = data["frames"]
    return [_png(f, "L") for f in frames], frames.shape[1], frames.shape[2]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


@lru_cache(maxsize=2048)
def _result(model_id: str, key: str, stamp: float) -> dict | None:
    path = catalog.result_file(model_id, key)
    if not stamp:
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def result(model_id: str | None, key: str) -> dict | None:
    if not model_id:
        return None
    return _result(model_id, key, _mtime(catalog.result_file(model_id, key)))


@lru_cache(maxsize=4)
def _flow_overlays(model_id: str, key: str, stamp: float) -> list[str]:
    """Flow per frame pair as an RGBA PNG: hue = direction, brightness = magnitude."""
    flow = np.load(catalog.flow_file(model_id, key)).astype(np.float32)  # (n-1, h/2, w/2, 2)
    with np.load(catalog.clip_file(key), allow_pickle=False) as data:
        sector = data["frames"].max(axis=0)[::2, ::2] > 0                 # nothing outside the image
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    inside = magnitude[:, sector[:magnitude.shape[1], :magnitude.shape[2]]]
    low, top = float(np.percentile(inside, 50)), max(float(np.percentile(inside, 99)), 1e-3)
    overlays = []
    for f, m in zip(flow, magnitude):
        hue = (np.arctan2(f[..., 1], f[..., 0]) + np.pi) / (2 * np.pi)
        rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1))
        # Only clear motion shows: transparent below the median, opaque-ish near the 99th percentile
        alpha = np.clip((m - low) / max(top - low, 1e-3), 0, 1) * 0.6
        alpha[~sector[:m.shape[0], :m.shape[1]]] = 0
        overlays.append(_png((np.dstack([rgb, alpha]) * 255).astype(np.uint8), "RGBA"))
    return overlays


def flow_overlays(model_id: str, key: str) -> list[str] | None:
    path = catalog.flow_file(model_id, key)
    return _flow_overlays(model_id, key, _mtime(path)) if path.is_file() else None


# -- background jobs ----------------------------------------------------------

_job: dict = {"thread": None, "label": "", "progress": []}


def _run_chain(commands, progress_files):
    env = {**os.environ, "MKL_THREADING_LAYER": "TBB"}
    for command in commands:
        subprocess.run(command, cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_job(label: str, models: list[str], clips: list[str] | None, flows: bool) -> bool:
    if _job["thread"] is not None and _job["thread"].is_alive():
        return False
    JOBS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    commands, files = [], []
    for model in dict.fromkeys(m for m in models if m):
        progress = JOBS / f"{stamp}_{model}.json"
        command = [sys.executable, "-m", "motion_estimator.viewer.worker", "--model", model,
                   "--progress", str(progress)]
        command += ["--all"] if clips is None else ["--clips", *clips]
        if flows:
            command.append("--flows")
        commands.append(command)
        files.append(progress)
    _job.update(label=label, progress=files,
                thread=threading.Thread(target=_run_chain, args=(commands, files), daemon=True))
    _job["thread"].start()
    return True


def job_status() -> tuple[str, bool]:
    """A one-line status, and whether a job is running."""
    running = _job["thread"] is not None and _job["thread"].is_alive()
    if not _job["progress"]:
        return "", False
    parts = []
    for path in _job["progress"]:
        if path.is_file():
            p = json.loads(path.read_text(encoding="utf-8"))
            parts.append(f"{p['model']}: {p['done']}/{p['total']}" + (" done" if p["state"] == "done" else ""))
        else:
            parts.append(f"{path.stem.split('_', 2)[-1]}: waiting")
    return f"{_job['label']} - " + " | ".join(parts) + ("" if running else " - finished"), running


# -- table and figures --------------------------------------------------------

def table_rows(model_a, model_b, dataset, view) -> list[dict]:
    rows = []
    for key in catalog.clip_keys():
        ds, patient, v = key.split("/")
        if (dataset != "all" and ds != dataset) or (view != "all" and v != view):
            continue
        row = {"clip": key, "dataset": ds.split("_")[0], "patient": patient, "view": v}
        for slot, model in (("a", model_a), ("b", model_b)):
            r = result(model, key)
            epe = float(r["epe_mm"]) if r is not None else np.nan
            row[f"epe_{slot}"] = None if not np.isfinite(epe) else round(epe, 3)
            row[f"seen_{slot}"] = "trained on" if r is not None and bool(r["seen"]) else ""
        if row["epe_a"] is not None and row["epe_b"] is not None:
            row["diff"] = round(row["epe_b"] - row["epe_a"], 3)
        rows.append(row)
    return rows


def _empty(message: str, height: int) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"color": INK_2})
    fig.update_layout(height=height, xaxis={"visible": False}, yaxis={"visible": False},
                      plot_bgcolor="white", margin={"l": 10, "r": 10, "t": 10, "b": 10})
    return fig


def frame_figure(key, frame, model_a, model_b, overlay, shown) -> go.Figure:
    if not key:
        return _empty("Select a patient on the left", 560)
    pngs, height, width = clip_frames(key)
    frame = min(frame, len(pngs) - 1)
    fig = go.Figure(go.Image(source=pngs[frame], hoverinfo="skip"))
    source = {"flow_a": model_a, "flow_b": model_b}.get(overlay)
    if source:
        overlays = flow_overlays(source, key)
        if overlays and frame < len(overlays):
            fig.add_trace(go.Image(source=overlays[frame], dx=2, dy=2, hoverinfo="skip"))
    for name, model, colour, dash_style in (("MITT", catalog.MITT, MITT_COLOUR, "dash"),
                                            ("A", model_a, A_COLOUR, "solid"),
                                            ("B", model_b, B_COLOUR, "solid")):
        r = result(model, key)
        if name not in shown or r is None or frame >= len(r["track_x"]):
            continue
        fig.add_trace(go.Scatter(x=r["track_x"][frame] - 1, y=r["track_y"][frame] - 1, mode="lines",
                                 line={"color": colour, "width": 2, "dash": dash_style},
                                 name=name if name == "MITT" else f"{name}: {model}",
                                 hovertemplate=f"{name}<extra></extra>"))
    fig.update_layout(height=560, margin={"l": 0, "r": 0, "t": 0, "b": 0}, uirevision=key,
                      legend={"orientation": "h", "y": 1.0, "x": 0, "bgcolor": "rgba(255,255,255,0.7)"},
                      xaxis={"visible": False, "range": [0, width]},
                      yaxis={"visible": False, "range": [height, 0], "scaleanchor": "x"})
    return fig


def curve_figures(key, frame, model_a, model_b) -> tuple[go.Figure, go.Figure]:
    if not key:
        return _empty("", 260), _empty("", 260)
    epe, strain = go.Figure(), go.Figure()
    mitt = result(catalog.MITT, key)
    if mitt is not None:
        n = len(mitt["strain"])
        strain.add_trace(go.Scatter(x=np.arange(1, n + 1), y=mitt["strain"], name="MITT",
                                    line={"color": INK, "width": 2, "dash": "dash"}))
        strain.add_hline(y=float(mitt["gls_ge"]), line={"color": INK_2, "dash": "dot", "width": 1},
                         annotation_text=f"GE GLS {float(mitt['gls_ge']):.1f}%", annotation_position="bottom right")
        strain.add_vline(x=int(mitt["es_frame"]), line={"color": INK_2, "width": 1},
                         annotation_text="ES", annotation_position="top")
    for name, model, colour in (("A", model_a, A_COLOUR), ("B", model_b, B_COLOUR)):
        r = result(model, key)
        if r is None:
            continue
        frames = np.arange(1, len(r["strain"]) + 1)
        label = f"{name}: {model}"
        epe.add_trace(go.Scatter(x=frames, y=r["epe_per_frame"], name=label, mode="lines+markers",
                                 marker={"size": 5}, line={"color": colour, "width": 2}))
        strain.add_trace(go.Scatter(x=frames, y=r["strain"], name=label,
                                    line={"color": colour, "width": 2}))
    for fig, title, unit in ((epe, "Contour EPE vs MITT per frame", "mm"), (strain, "Longitudinal strain", "%")):
        fig.add_vline(x=frame + 1, line={"color": "#9a9993", "width": 1, "dash": "dot"})
        fig.update_layout(height=260, title={"text": title, "x": 0.01, "font": {"size": 13}},
                          margin={"l": 50, "r": 10, "t": 36, "b": 36}, plot_bgcolor="white",
                          legend={"orientation": "h", "y": -0.25, "font": {"size": 10}},
                          xaxis={"title": "frame", "gridcolor": GRID}, yaxis={"title": unit, "gridcolor": GRID})
    return epe, strain


def summary(rows) -> tuple[list, go.Figure]:
    pairs = [(r["epe_a"], r["epe_b"]) for r in rows if r["epe_a"] is not None and r["epe_b"] is not None]
    lines = []
    for slot, colour in (("a", A_COLOUR), ("b", B_COLOUR)):
        values = np.array([r[f"epe_{slot}"] for r in rows if r[f"epe_{slot}"] is not None], float)
        text = (f"{values.mean():.2f} +- {values.std(ddof=1):.2f} mm (n={len(values)})"
                if len(values) > 1 else "not computed")
        lines.append(html.Div([html.Span("● ", style={"color": colour}),
                               html.B(f"{slot.upper()}: "), text]))
    if pairs:
        a, b = np.array(pairs).T
        lines.append(html.Div(f"B better on {(b < a).sum()}/{len(pairs)} clips, mean B - A {np.mean(b - a):+.2f} mm",
                              style={"color": INK_2}))
    fig = go.Figure()
    if pairs:
        top = float(max(max(a), max(b))) * 1.05
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line={"color": GRID}, hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=a, y=b, mode="markers", marker={"size": 7, "color": INK_2, "opacity": 0.6},
                                 hovertemplate="A %{x:.2f} / B %{y:.2f} mm<extra></extra>", showlegend=False))
    fig.update_layout(height=230, margin={"l": 45, "r": 10, "t": 10, "b": 40}, plot_bgcolor="white",
                      xaxis={"title": "A EPE (mm)", "gridcolor": GRID}, yaxis={"title": "B EPE (mm)", "gridcolor": GRID})
    return lines, fig


# -- layout -------------------------------------------------------------------

def _model_options():
    return [{"label": m.label, "value": m.id} for m in catalog.models().values()]


def layout():
    options = _model_options()
    first = [o["value"] for o in options]
    columns = [{"name": "dataset", "id": "dataset"}, {"name": "patient", "id": "patient"},
               {"name": "view", "id": "view"},
               {"name": "A EPE (mm)", "id": "epe_a", "type": "numeric"},
               {"name": "B EPE (mm)", "id": "epe_b", "type": "numeric"},
               {"name": "B - A", "id": "diff", "type": "numeric"},
               {"name": "A", "id": "seen_a"}, {"name": "B", "id": "seen_b"}]
    controls = dbc.Card(dbc.CardBody([
        dbc.Label("Model A", style={"color": A_COLOUR, "fontWeight": 600}),
        dcc.Dropdown(options, first[0] if first else None, id="model-a", clearable=False),
        dbc.Label("Model B", className="mt-2", style={"color": B_COLOUR, "fontWeight": 600}),
        dcc.Dropdown(options, first[1] if len(first) > 1 else None, id="model-b"),
        html.Div([dbc.Button("Predict selected clip", id="predict-one", size="sm", className="me-2"),
                  dbc.Button("Run A & B on all clips", id="predict-all", size="sm", outline=True)],
                 className="mt-3"),
        html.Div(id="job-status", className="mt-2 small", style={"color": INK_2}),
    ]), className="mb-3")
    patients = dbc.Card(dbc.CardBody([
        dbc.Row([dbc.Col(dcc.Dropdown([{"label": "All datasets", "value": "all"},
                                       {"label": "Retest", "value": "Retest_dataset"},
                                       {"label": "Test", "value": "Test_dataset"}], "all", id="dataset",
                                      clearable=False)),
                 dbc.Col(dcc.Dropdown([{"label": "2ch + 4ch", "value": "all"}, {"label": "2ch", "value": "2ch"},
                                       {"label": "4ch", "value": "4ch"}], "all", id="view", clearable=False))]),
        dash_table.DataTable(id="clips", columns=columns, data=[], row_selectable="single", sort_action="native",
                             page_size=14, style_table={"marginTop": "8px"},
                             style_cell={"fontSize": 12, "padding": "3px 6px", "fontFamily": "inherit"},
                             style_header={"fontWeight": 600}),
    ]), className="mb-3")
    summary_card = dbc.Card(dbc.CardBody([html.Div(id="summary-text", className="small"),
                                          dcc.Graph(id="summary-plot", config={"displayModeBar": False})]))
    player = dbc.Card(dbc.CardBody([
        html.Div(id="clip-title", className="fw-semibold mb-2"),
        dcc.Graph(id="frame", config={"displaylogo": False}),
        dbc.Row([
            dbc.Col(dbc.Button("Play", id="play", size="sm", color="primary"), width="auto"),
            dbc.Col(dcc.Slider(id="frame-slider", min=0, max=1, step=1, value=0, marks=None,
                               tooltip={"placement": "bottom"})),
            dbc.Col(dcc.Dropdown([{"label": f"{f} fps", "value": f} for f in (2, 5, 10)], 5, id="fps",
                                 clearable=False, style={"width": "90px"}), width="auto"),
        ], align="center", className="mt-2"),
        dbc.Row([
            dbc.Col(dcc.Checklist([{"label": " MITT", "value": "MITT"}, {"label": " A", "value": "A"},
                                   {"label": " B", "value": "B"}], ["MITT", "A", "B"], id="shown", inline=True,
                                  inputStyle={"marginLeft": "10px"})),
            dbc.Col(dcc.RadioItems([{"label": " no flow", "value": "none"}, {"label": " flow A", "value": "flow_a"},
                                    {"label": " flow B", "value": "flow_b"}], "none", id="overlay", inline=True,
                                   inputStyle={"marginLeft": "10px"}), width="auto"),
        ], className="mt-2 small"),
    ]), className="mb-3")
    curves = dbc.Row([dbc.Col(dcc.Graph(id="epe-curve", config={"displayModeBar": False}), md=6),
                      dbc.Col(dcc.Graph(id="strain-curve", config={"displayModeBar": False}), md=6)])
    return dbc.Container([
        dbc.NavbarSimple(brand="Cardiac motion viewer - in vivo dataset", color="dark", dark=True,
                         className="mb-3", fluid=True),
        dbc.Row([dbc.Col([controls, patients, summary_card], lg=4),
                 dbc.Col([player, curves], lg=8)]),
        dcc.Store(id="clip"), dcc.Store(id="results-version", data=0),
        dcc.Interval(id="tick", interval=200, disabled=True),
        dcc.Interval(id="job-poll", interval=2000),
    ], fluid=True)


# -- callbacks ----------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Cardiac motion viewer")
app.layout = layout  # a function, so new models appear on reload
_state = {"version": 0}


@app.callback(Output("job-status", "children"), Output("results-version", "data"),
              Input("predict-one", "n_clicks"), Input("predict-all", "n_clicks"), Input("job-poll", "n_intervals"),
              State("clip", "data"), State("model-a", "value"), State("model-b", "value"),
              prevent_initial_call=True)
def jobs(_one, _all, _poll, clip, model_a, model_b):
    if ctx.triggered_id == "predict-one" and clip:
        if not start_job(f"Predicting {clip}", [catalog.MITT, model_a, model_b], [clip], flows=True):
            return "A job is already running", dash.no_update
    elif ctx.triggered_id == "predict-all":
        if not start_job("All clips", [catalog.MITT, model_a, model_b], None, flows=False):
            return "A job is already running", dash.no_update
    status, running = job_status()
    version = sum(json.loads(p.read_text(encoding="utf-8"))["done"] for p in _job["progress"] if p.is_file())
    version += 0 if running else 10 ** 6
    if version == _state["version"]:
        return status, dash.no_update
    _state["version"] = version
    return status, version


@app.callback(Output("clips", "data"), Output("clips", "selected_rows"),
              Input("model-a", "value"), Input("model-b", "value"), Input("dataset", "value"),
              Input("view", "value"), Input("results-version", "data"), State("clips", "selected_rows"),
              State("clips", "data"), State("clip", "data"))
def refresh_table(model_a, model_b, dataset, view, _version, selected, old, clip):
    rows = table_rows(model_a, model_b, dataset, view)
    keys = [r["clip"] for r in rows]
    return rows, ([keys.index(clip)] if clip in keys else [])


@app.callback(Output("clip", "data"), Output("frame-slider", "max"), Output("frame-slider", "value"),
              Output("clip-title", "children"),
              Input("clips", "selected_rows"), State("clips", "data"), State("clip", "data"))
def select_clip(selected, rows, current):
    if not selected or not rows:
        return dash.no_update, dash.no_update, dash.no_update, "No patient selected"
    key = rows[selected[0]]["clip"]
    if key == current:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    pngs, _, _ = clip_frames(key)
    mitt = result(catalog.MITT, key)
    es = f", ES frame {int(mitt['es_frame'])}" if mitt is not None else ""
    return key, len(pngs) - 1, 0, f"{key.replace('_dataset', '')} - {len(pngs)} frames{es}"


@app.callback(Output("tick", "disabled"), Output("play", "children"), Output("tick", "interval"),
              Input("play", "n_clicks"), Input("fps", "value"), State("tick", "disabled"),
              prevent_initial_call=True)
def play(_clicks, fps, disabled):
    if ctx.triggered_id == "play":
        disabled = not disabled
    return disabled, ("Play" if disabled else "Pause"), int(1000 / fps)


@app.callback(Output("frame-slider", "value", allow_duplicate=True), Input("tick", "n_intervals"),
              State("frame-slider", "value"), State("frame-slider", "max"), prevent_initial_call=True)
def advance(_n, frame, last):
    return 0 if frame >= (last or 0) else frame + 1


@app.callback(Output("frame", "figure"),
              Input("clip", "data"), Input("frame-slider", "value"), Input("model-a", "value"),
              Input("model-b", "value"), Input("overlay", "value"), Input("shown", "value"),
              Input("results-version", "data"))
def draw_frame(clip, frame, model_a, model_b, overlay, shown, _version):
    return frame_figure(clip, frame or 0, model_a, model_b, overlay, shown or [])


@app.callback(Output("epe-curve", "figure"), Output("strain-curve", "figure"),
              Input("clip", "data"), Input("frame-slider", "value"), Input("model-a", "value"),
              Input("model-b", "value"), Input("results-version", "data"))
def draw_curves(clip, frame, model_a, model_b, _version):
    return curve_figures(clip, frame or 0, model_a, model_b)


@app.callback(Output("summary-text", "children"), Output("summary-plot", "figure"), Input("clips", "data"))
def draw_summary(rows):
    return summary(rows or [])


def main() -> int:
    app.run(host="127.0.0.1", port=int(os.environ.get("VIEWER_PORT", 8050)), debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
