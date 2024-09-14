import ast
from pathlib import Path
import concurrent.futures
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from pattern_recognition.utils.parameters_dics import parameters


def process_trace(i, label, traces_array, color_dict, cardiac_events, min_height, max_height):
    row = (i + 1)
    col = 1

    max_trace = np.nanmax(traces_array, axis=0)
    min_trace = np.nanmin(traces_array, axis=0)
    mean_trace = np.nanmean(traces_array, axis=0)

    events = []
    if cardiac_events is not None:
        for idx, (x, event) in enumerate(zip(cardiac_events['event_time'].values, cardiac_events['event_name'].values)):
            events.append((x, event, min_height, max_height))

    return row, col, label, max_trace, min_trace, mean_trace, color_dict.get(label, 'blue'), events


def prepare_traces(traces, isInterp):
    if isInterp == 'Reconstructed':
        traces_array = traces
    else:
        traces_array = [np.array(ast.literal_eval(trace)) for trace in traces]

    max_length = max(len(trace) for trace in traces_array)
    traces_array = np.array(
        [np.pad(trace, (0, max_length - len(trace)), 'constant', constant_values=np.nan) for trace in traces_array])

    return traces_array


class TracePlotter:
    def __init__(self, tracesDF, cardiac_events, concat_traces_by):
        self.traces_df = None
        self.tracesDF = tracesDF
        self.cardiac_events = cardiac_events
        self.concat_traces_by = concat_traces_by

    def group_traces(self, column_name):
        dropna_tracesDF = self.tracesDF.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[
            ~(dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0) if x is not None else True))]
        grouped = dropna_tracesDF.groupby('study_nr')
        valid_study_df = grouped.filter(lambda x: len(x) == 18)

        valid_study_df[column_name] = valid_study_df[column_name].str.strip()
        valid_study_df[column_name] = valid_study_df[column_name].str.strip().str.lower()
        self.traces_df = valid_study_df.groupby(column_name)['strain_trace'].apply(list)
        pass

    def plot_trace_(self, plot_type, color_dict, isInterp, fig_height, fig_width, save_plot):
        labels = self.traces_df.keys()

        if plot_type == 'Segment':
            fig = make_subplots(rows=(len(labels) + 1), cols=1, vertical_spacing=0.01,
                                row_heights=[400 for _ in range(len(labels) + 1)])
        else:
            fig = make_subplots(rows=(len(labels) + 1), cols=1, vertical_spacing=0.03)

        interp_text = isInterp
        concatenate_tex = self.concat_traces_by if isInterp == 'Reconstructed' else ''

        traces_array = {label: prepare_traces(traces, isInterp) for label, traces in self.traces_df.items()}

        min_height = min(np.nanmin(arr) for arr in traces_array.values())
        max_height = max(np.nanmax(arr) for arr in traces_array.values())

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_trace, i, label, traces, color_dict, self.cardiac_events, min_height,
                                       max_height)
                       for i, (label, traces) in enumerate(traces_array.items(), start=1)]

            for future in concurrent.futures.as_completed(futures):
                row, col, label, max_trace, min_trace, mean_trace, color, events = future.result()

                fig.add_trace(go.Scatter(y=max_trace, mode='lines', line=dict(width=0, color=color),
                                         fill=None, showlegend=False), row=row, col=col)
                fig.add_trace(go.Scatter(y=min_trace, mode='lines', line=dict(width=0, color=color),
                                         fill='tonexty', showlegend=False), row=row, col=col)
                fig.add_trace(go.Scatter(y=mean_trace, mode='lines', line=dict(width=2, color=color)),
                              row=row, col=col)
                fig.update_xaxes(title_text=f'<b>{label}</b>', row=row, col=col)

                for x, event, min_height, max_height in events:
                    fig.add_shape(type="line", x0=x, y0=min_height, x1=x, y1=max_height, yref="paper",
                                  line=dict(color="Red", dash="dash", width=1), row=row, col=col)
                    fig.add_annotation(x=x, y=max_height, yref="paper", text=event, showarrow=False,
                                       font=dict(size=10, color="Black", family="Arial, bold"), row=row, col=col)

        fig.update_layout(width=fig_width, height=fig_height,
                          title={'text': f'<b>{concatenate_tex} {interp_text} Strain Traces per {plot_type}</b>',
                                 'x': 0.5, 'xanchor': 'center', 'y': 0.99, 'yanchor': 'top'})

        fig['layout']['yaxis']['automargin'] = True
        fig['layout']['yaxis2']['automargin'] = True
        fig.update_layout(autosize=True, margin=dict(t=50, b=50, l=50, r=50), showlegend=False, plot_bgcolor='white')

        if save_plot:
            dir_path = Path(parameters['save_plot_dir'])
            file_name = f"{self.concat_traces_by}{isInterp}_{plot_type}.html"
            file_Path = dir_path / file_name

            pio.write_html(fig, str(file_Path))

        return fig


    def plot_trace(self, plot_type, color_dict, isInterp, fig_height, fig_width, save_plot):
        labels = self.traces_df.keys()

        if plot_type == 'Segment':
            fig = make_subplots(rows=(len(labels) + 1), cols=1, vertical_spacing=0.01,
                                row_heights=[400 for _ in range(len(labels) + 1)])
        else:
            fig = make_subplots(rows=(len(labels) + 1), cols=1, vertical_spacing=0.03)

        interp_text = isInterp
        concatenate_tex = self.concat_traces_by if isInterp == 'Reconstructed' else ''
        for i, (label, traces) in enumerate(self.traces_df.items(), start=1):
            label = label.lower()
            color_dict = {k.lower(): v for k, v in color_dict.items()}
            row = (i + 1)
            col = 1
            if isInterp == 'Reconstructed':
                traces_array = traces
            else:
                traces_array = [np.array(ast.literal_eval(trace)) for trace in traces]

            max_length = max(len(trace) for trace in traces_array)

            traces_array = [np.pad(trace, (0, max_length - len(trace)), 'constant', constant_values=np.nan) for trace in
                            traces_array]

            max_trace = np.nanmax(traces_array, axis=0)
            min_trace = np.nanmin(traces_array, axis=0)
            fig.add_trace(go.Scatter(y=max_trace, mode='lines', line=dict(width=0, color=color_dict.get(label)),
                                     fill=None, showlegend=False), row=row, col=col)
            fig.add_trace(go.Scatter(y=min_trace, mode='lines', line=dict(width=0, color=color_dict.get(label)),
                                     fill='tonexty', showlegend=False), row=row, col=col)
            mean_trace = np.nanmean(traces_array, axis=0)

            fig.add_trace(
                go.Scatter(y=mean_trace, mode='lines', line=dict(width=2, color=color_dict.get(label))),
                row=row, col=col)
            fig.update_xaxes(title_text=f'<b>{label}</b>', row=row, col=col)

            min_height = np.nanmin(traces_array)
            max_height = np.nanmax(traces_array)

            if self.cardiac_events is not None and isInterp == 'Interpolated' or isInterp == 'Reconstructed':
                for idx, (x, event) in enumerate(
                        zip(self.cardiac_events['event_time'].values, self.cardiac_events['event_name'].values)):
                    fig.add_shape(
                        type="line",
                        x0=x, y0=min_height, x1=x, y1=max_height, yref="paper",
                        line=dict(color="Red", dash="dash", width=1),
                        row=row, col=col
                    )
                    fig.add_annotation(
                        x=x, y=max_height, yref="paper", text=event, showarrow=False,
                        font=dict(size=10, color="Black", family="Arial, bold"),
                        row=row, col=col
                    )
        fig.update_layout(width=fig_width, height=fig_height,
                          title={'text': f'<b>{concatenate_tex} {interp_text} Strain Traces per {plot_type}</b>',
                                 'x': 0.5, 'xanchor': 'center', 'y': 0.99, 'yanchor': 'top'})

        fig['layout']['yaxis']['automargin'] = True
        fig['layout']['yaxis2']['automargin'] = True
        fig.update_layout(autosize=True, margin=dict(t=50, b=50, l=50, r=50), showlegend=False, plot_bgcolor='white')

        if save_plot:
            dir_path = Path(parameters['save_plot_dir'])
            file_name = f"{self.concat_traces_by}{isInterp}_{plot_type}.html"
            file_Path = dir_path / file_name

            pio.write_html(fig, str(file_Path))

        return fig
