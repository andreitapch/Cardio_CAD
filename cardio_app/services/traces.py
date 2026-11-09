import numpy as np
import pandas as pd

LABEL_COLUMNS = ["segment_wall", "curve_segment", "cardiac_view"]
SEGMENTS_PER_STUDY = 18


def parse(trace):
    if isinstance(trace, str):
        return np.fromstring(trace.strip().strip("[]"), sep=",")
    return np.asarray(trace, dtype=float)


def parse_rows(frame):
    frame = frame.copy()
    for column in LABEL_COLUMNS + ["id_position"]:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[~frame["segment_wall"].str.lower().eq("apex") & ~frame["curve_segment"].isin(["0", "Apex"])]
    frame["trace"] = frame["strain_trace"].map(parse)
    frame = frame[frame["trace"].map(lambda t: t.size > 0 and not np.all(t == 0))]
    frame["position"] = frame["id_position"].astype(int)
    return frame.reset_index(drop=True)


def complete(frame):
    counts = frame.groupby("study_nr")["trace"].transform("size")
    return frame[counts == SEGMENTS_PER_STUDY].reset_index(drop=True)


def clean(frame):
    return complete(parse_rows(frame))


def padded(traces):
    length = max(t.size for t in traces)
    return np.array([np.pad(t, (0, length - t.size), constant_values=np.nan) for t in traces])


def grouped(frame, by):
    labels = frame[by].str.lower()
    return {label: padded(list(group["trace"])) for label, group in frame.groupby(labels)}


def event_marks(events, order):
    events = events[events["event_name"].str.strip().isin(order)]
    means = events.groupby(events["event_name"].str.strip())["event_time"].mean().round()
    return [(float(means[name]), name) for name in order if name in means]


def study(frame, study_nr):
    rows = frame[frame["study_nr"] == study_nr]
    return pd.DataFrame({"label": rows["curve_segment"].to_numpy(), "trace": rows["trace"].to_numpy()})


def ordered(frame):
    if "position" not in frame:
        frame = frame.assign(position=frame["id_position"].astype(str).str.strip().astype(int))
    return frame.sort_values(["study_nr", "position"]).reset_index(drop=True)


def matrix(frame):
    frame = ordered(frame)
    groups = frame.groupby("study_nr", sort=False)
    studies = np.array(list(groups.groups))
    lengths = groups["trace"].apply(lambda t: [v.size for v in t])
    return studies, np.stack([np.concatenate(list(group["trace"])) for _, group in groups]), lengths


def split(vector, lengths):
    return np.split(vector, np.cumsum(lengths)[:-1])
