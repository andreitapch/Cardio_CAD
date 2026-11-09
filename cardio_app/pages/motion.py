import streamlit as st

from cardio_app import state, ui
from cardio_app.data import motion
from cardio_app.jobs import motion as jobs
from cardio_app.plots import motion as plots
from cardio_app.services import motion as service

PAGE = "motion"
DATASETS = {"All datasets": "all", "Retest": "Retest_dataset", "Test": "Test_dataset"}
VIEWS = {"2ch + 4ch": "all", "2ch": "2ch", "4ch": "4ch"}
COLUMNS = {"dataset": "dataset", "patient": "patient", "view": "view", "epe_a": "A EPE (mm)",
           "epe_b": "B EPE (mm)", "diff": "B - A", "seen_a": "A", "seen_b": "B"}


def contours_for(key, frame, model_a, model_b, shown):
    rows = []
    for name, model, colour, dashed in (("MITT", motion.MITT, plots.MITT_COLOUR, True),
                                        ("A", model_a, plots.A_COLOUR, False), ("B", model_b, plots.B_COLOUR, False)):
        r = motion.result(model, key)
        if name in shown and r is not None and frame < len(r["track_x"]):
            rows.append((r["track_x"][frame], r["track_y"][frame], colour, dashed))
    return rows


def overlay_for(key, frame, model):
    flow = motion.flow(model, key) if model else None
    if flow is None or frame >= len(flow):
        return None
    low, top, sector = service.flow_scale(flow, motion.frames(key))
    return service.flow_rgba(flow[frame], sector, low, top)


def model_results(key, model_a, model_b):
    rows = []
    for name, model, colour in (("A", model_a, plots.A_COLOUR), ("B", model_b, plots.B_COLOUR)):
        r = motion.result(model, key)
        if r is not None:
            rows.append((f"{name}: {model}", colour, r))
    return rows


st.title("Cardiac motion viewer")
models = motion.models()
ids = list(models)
left, right = st.columns((2, 3))

with left:
    model_a = st.selectbox("Model A", ids, format_func=models.get, key=state.key(PAGE, "model_a"))
    model_b = st.selectbox("Model B", [None, *ids], index=min(2, len(ids)),
                           format_func=lambda m: models.get(m, "none"), key=state.key(PAGE, "model_b"))
    clip = state.load(PAGE, "clip")
    job = state.load(PAGE, "job")
    busy = job is not None and not job.finished()
    one, every = st.columns(2)
    if one.button("Predict selected clip", disabled=clip is None or busy):
        ui.launch(PAGE, jobs.start(f"Predicting {clip}", [motion.MITT, model_a, model_b], [clip], True))
    if every.button("Run A & B on all clips", disabled=busy):
        ui.launch(PAGE, jobs.start("All clips", [motion.MITT, model_a, model_b], None, False))
    ui.watch(PAGE)
    dataset_col, view_col = st.columns(2)
    dataset = DATASETS[dataset_col.selectbox("Dataset", list(DATASETS), key=state.key(PAGE, "dataset"))]
    view = VIEWS[view_col.selectbox("View", list(VIEWS), key=state.key(PAGE, "view"))]
    table = motion.table(model_a, model_b, dataset, view)
    event = st.dataframe(table.drop(columns="clip").rename(columns=COLUMNS), hide_index=True, height=420,
                         on_select="rerun", selection_mode="single-row", key=state.key(PAGE, "table"))
    if event.selection.rows:
        clip = state.store(PAGE, "clip", table.iloc[event.selection.rows[0]]["clip"])
    lines, pairs = service.summary(table)
    for line in lines:
        st.write(line)
    st.pyplot(plots.epe_scatter(pairs))

with right:
    if clip is None:
        st.info("Select a patient in the table")
        st.stop()
    frames = motion.frames(clip)
    mitt = motion.result(motion.MITT, clip)
    es = f", ES frame {int(mitt['es_frame'])}" if mitt is not None else ""
    st.subheader(f"{clip.replace('_dataset', '')} - {len(frames)} frames{es}")
    play_col, fps_col, shown_col, overlay_col = st.columns((1, 1, 2, 2))
    playing = play_col.toggle("Play", key=state.key(PAGE, "playing"))
    fps = fps_col.selectbox("fps", (2, 5, 10), index=1, key=state.key(PAGE, "fps"))
    shown = shown_col.multiselect("Contours", ["MITT", "A", "B"], default=["MITT", "A", "B"], key=state.key(PAGE, "shown"))
    overlay = overlay_col.radio("Flow overlay", ["none", "A", "B"], horizontal=True, key=state.key(PAGE, "overlay"))
    if state.load(PAGE, "frames_clip") != clip:
        state.store(PAGE, "frames_clip", clip)
        st.session_state[state.key(PAGE, "frame")] = 0

    @st.fragment(run_every=1 / fps if playing else None)
    def player():
        last = len(frames) - 1
        if playing:
            current = st.session_state.get(state.key(PAGE, "frame"), 0)
            st.session_state[state.key(PAGE, "frame")] = 0 if current >= last else current + 1
        frame = st.slider("Frame", 0, last, key=state.key(PAGE, "frame"))
        source = {"A": model_a, "B": model_b}.get(overlay)
        image = plots.composite(frames[frame], overlay_for(clip, frame, source),
                                contours_for(clip, frame, model_a, model_b, shown))
        st.image(image, width="stretch")
        results = model_results(clip, model_a, model_b)
        epe_col, strain_col = st.columns(2)
        epe_col.pyplot(plots.epe_curve(frame, results))
        strain_col.pyplot(plots.strain_curve(frame, mitt, results))

    player()
