import numpy as np
import pandas as pd
def interpolate_array(coords, idxToSplit, newIdxFirPart, newIdxSecPart):
    interpArray = None
    traceFirstPart = coords[:idxToSplit]
    traceSecondPart = coords[idxToSplit:]

    x = np.arange(len(traceFirstPart))
    traceInterp1 = np.linspace(0, len(traceFirstPart) - 1, newIdxFirPart)
    trace1Interp = np.interp(traceInterp1, x, traceFirstPart)

    y_origin = np.arange(len(traceSecondPart))
    traceInterp2 = np.linspace(0, len(traceSecondPart) - 1, newIdxSecPart)
    trace2Interp = np.interp(traceInterp2, y_origin, traceSecondPart)

    interpArray = np.concatenate((trace1Interp, trace2Interp))
    interpArray = np.reshape(interpArray, (1, -1))
    return interpArray


def event_loader(cardiac_events_order, cardiac_events, interpSelection):
    filtered_cardiac_ev = cardiac_events[
        cardiac_events['event_name'].isin(cardiac_events_order)].reset_index(
        drop=True)
    filtered_cardiac_ev['event_name'] = pd.Categorical(filtered_cardiac_ev['event_name'],
                                                       categories=cardiac_events_order, ordered=True)
    filtered_cardiac_ev = filtered_cardiac_ev.sort_values(by=['study_nr', 'segment_wall', 'event_name'])
    grouped_ev = filtered_cardiac_ev.groupby('event_name', observed=False)

    if interpSelection == 'max':
        cardiac_evDF = grouped_ev[['event_length', 'event_time']].max().reset_index(
        ).set_index('event_name').loc[cardiac_events_order].reset_index()
    else:
        cardiac_evDF = grouped_ev[['event_length', 'event_time']].mean().round().astype(
            int).reset_index().set_index('event_name').loc[cardiac_events_order].reset_index()

    return cardiac_evDF, filtered_cardiac_ev