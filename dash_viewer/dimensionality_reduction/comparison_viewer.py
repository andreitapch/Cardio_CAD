import ast

from dash import dcc, html, Input, Output, dash, State, callback, ctx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import torch
from sklearn.decomposition import PCA

from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.utils.plot_dics import segment_color_dict
from pattern_recognition.data.loadTraces import LoadTraces
from dash_viewer.dimensionality_reduction.AutoencoderDm import (calculate_autoencoder,
                                                                reconstruct_traces_autoencoder,
                                                                mean_squared_error,
                                                                explained_variance, split_train_val)
from dash_viewer.dimensionality_reduction.pca_viewer import (calculate_pca, holdout_r2,
                                                             reconstruct_traces_pca,
                                                             r2_score_reconstruction)
from dash_viewer.session_state import RunStore, new_session_id

# Same per-session run state as the other reduction tabs: a fitted model plus
# every trace in the cohort cannot go through dcc.Store.
_runs = RunStore()

# Kept away from the PCA/Autoencoder tab's "runs/autoencoder_experiment", which
# the trainer wipes on every run - a comparison must not destroy that tab's logs.
AE_LOG_DIR = 'runs/autoencoder_comparison'
AE_EPOCHS = 1000
# Mirrors calculate_autoencoder's defaults, so the slice scored here is exactly
# the slice the trainer held out.
VAL_FRACTION = 0.15
SPLIT_SEED = 42


@callback(
    [Output('cmp-study-dropdown', 'options'),
     Output('cmp-study-dropdown', 'value'),
     Output('cmp-metrics-table', 'children'),
     Output('cmp-status-txt', 'children')],
    [Input('cmp-run-button', 'n_clicks')],
    [State('cmp-data-source-dropdown', 'value'),
     State('cmp-concat-dropdown', 'value'),
     State('cmp-latent-slider', 'value'),
     State('cmp-session-id', 'data')]
)
def run_comparison(run_btn, data_source, concat_traces_by, latent_dim, session_id):
    if 'cmp-run-button' != ctx.triggered_id or not run_btn:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    data_source_mapping = {
        'Speqle': 'speqle_segmentalStrainInterp_T',
        'GE': 'st_segmentalStrainInterp_T',
        'Philips': 'phi_segmentalStrainInterp_T',
        'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
    }
    if data_source not in data_source_mapping:
        raise ValueError("Invalid data source provided")

    run = ComparisonViewer()
    run.data_source = data_source
    run.table_name = data_source_mapping[data_source]
    run.load_study_nr_dropdown()
    run.compare(concat_traces_by, latent_dim)

    # Published only once both models are fitted: a half-built run must never be
    # visible to the plotting callback.
    _runs.put(session_id, run)

    options = [{'label': i, 'value': i} for i in run.study_nr]
    # Cleared on every completed run, otherwise keeping the previous study_nr
    # leaves the plot callback untriggered and the figure keeps showing the
    # previous run's reconstructions under the new options.
    return options, None, build_metrics_table(run.metrics), run.status


@callback(
    [Output('cmp-graph', 'figure'),
     Output('cmp-error-txt', 'children')],
    [Input('cmp-study-dropdown', 'value'),
     State('cmp-session-id', 'data')]
)
def plot_comparison(study_nr, session_id):
    run = _runs.get(session_id)
    if study_nr is None or run is None or run.pca_tracesDF is None:
        return go.Figure(), ''

    original = curves_for_study(run.tracesDF, study_nr, 'strain_trace')
    pca_curves = curves_for_study(run.pca_tracesDF, study_nr, 'strain_trace')
    ae_curves = curves_for_study(run.ae_tracesDF, study_nr, 'strain_trace')

    fig = go.Figure()
    for (segment, curve), (_, pca_curve), (_, ae_curve) in zip(original, pca_curves, ae_curves):
        color = segment_color_dict.get(segment, 'grey')
        # One colour per segment, one dash pattern per method: the three curves
        # of a segment are meant to be read on top of each other.
        fig.add_trace(go.Scatter(y=curve, mode='lines', name=segment,
                                 legendgroup=segment,
                                 line=dict(color=color)))
        fig.add_trace(go.Scatter(y=pca_curve, mode='lines', name=f'{segment} PCA',
                                 legendgroup=segment, showlegend=False,
                                 line=dict(color=color, dash='dash')))
        fig.add_trace(go.Scatter(y=ae_curve, mode='lines', name=f'{segment} AE',
                                 legendgroup=segment, showlegend=False,
                                 line=dict(color=color, dash='dot')))

    fig.update_layout(
        title=f'Study {study_nr} - solid: original, dashed: PCA, dotted: Autoencoder '
              f'(latent dim {run.latent_dim})',
        title_x=0.5, autosize=False, height=650)

    pca_mse = study_mse(original, pca_curves)
    ae_mse = study_mse(original, ae_curves)
    readout = html.Div([
        html.Span(f'Study {study_nr} reconstruction MSE - '),
        html.Span(f'PCA {pca_mse:.4f}', style={'fontWeight': 'bold'}),
        html.Span(' | '),
        html.Span(f'Autoencoder {ae_mse:.4f}', style={'fontWeight': 'bold'}),
    ])
    return fig, readout


def build_metrics_table(metrics):
    if not metrics:
        return ''

    header = html.Thead(html.Tr([html.Th(col) for col in [
        'Method', 'Latent dim', 'Features in', 'n', 'Params',
        'MSE (in-sample)', 'R2 (in-sample)', 'R2 (train split)', 'R2 (held-out)',
        'Overfit gap', 'Cum. explained variance', 'Notes']]))

    rows = []
    for row in metrics:
        # train - held-out, the only column that says whether a score generalises.
        tr, ho = row.get('holdout_r2_train'), row.get('holdout_r2')
        gap = fmt(tr - ho) if tr is not None and ho is not None else 'n/a'
        rows.append(html.Tr([
            html.Td(row['method']),
            html.Td(row['latent_dim']),
            html.Td(row.get('n_features', 'n/a')),
            html.Td(row.get('n_samples', 'n/a')),
            html.Td(f"{row.get('n_params') or 0:,}"),
            html.Td(fmt(row['mse'])),
            html.Td(fmt(row['r2'])),
            html.Td(fmt(tr)),
            html.Td(fmt(ho)),
            html.Td(gap),
            html.Td(fmt(row['cum_var_ratio'])),
            html.Td(html.Small(row.get('detail', ''), className='text-muted')),
        ]))

    note = html.Small(
        'R2 = 1 - MSE/Var, so 0.00 is the mean-curve baseline and 1.00 is perfect '
        'reconstruction; MSE alone is not comparable across datasets because it is '
        'unnormalised. Held-out R2 is the number to report - in-sample R2 rises with '
        'every extra parameter. By Eckart-Young, PCA is the best possible LINEAR '
        'reconstruction at this rank, so it is the floor a nonlinear method must beat.',
        className='text-muted')

    return html.Div([
        dbc.Table([header, html.Tbody(rows)], bordered=True, striped=True, hover=True,
                  style={'width': '100%'}),
        note,
    ])


def fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n/a'
    return f'{value:.4f}'


def curves_for_study(df, study_nr, column):
    """(segment, curve) pairs for one study, ordered as the segments are stored."""
    study_df = df[df['study_nr'] == study_nr].sort_values(by='id_position').reset_index(drop=True)
    pairs = []
    for idx, raw in enumerate(study_df[column]):
        pairs.append((str(study_df['curve_segment'][idx]).strip(), as_curve(raw)))
    return pairs


def as_curve(raw):
    # Traces come back from the DB as a string literal; reconstructions are
    # already arrays.
    if isinstance(raw, str):
        return np.array(ast.literal_eval(raw), dtype=float)
    return np.asarray(raw, dtype=float)


def study_mse(original_pairs, reconstructed_pairs):
    errors = [np.mean((orig - rec) ** 2)
              for (_, orig), (_, rec) in zip(original_pairs, reconstructed_pairs)]
    return float(np.mean(errors)) if errors else float('nan')


def pca_metrics(array_curves, latent_dim):
    """Fit PCA at latent_dim and score it, reusing the tab's own reconstruction."""
    scores, model, n_used = calculate_pca(latent_dim, array_curves)
    reconstructed = reconstruct_traces_pca(scores, model)

    # Fitted on the autoencoder's own training half and scored on its held-out
    # half, so both rows of the table describe the same patients.
    split = shared_split_indices(len(array_curves))
    if split is None:
        held_out = holdout_r2(array_curves, n_used)
    else:
        val_idx, train_idx = split
        train, val = array_curves[train_idx], array_curves[val_idx]
        fold = PCA(n_components=min(n_used, len(train), array_curves.shape[1]),
                   svd_solver='full').fit(train)
        held_out = (r2_score_reconstruction(fold, train),
                    r2_score_reconstruction(fold, val), n_used)
    metrics = {
        'method': 'PCA (linear)',
        'latent_dim': n_used,
        'n_samples': int(array_curves.shape[0]),
        'n_features': int(array_curves.shape[1]),
        'mse': float(np.mean((array_curves - reconstructed) ** 2)),
        'r2': r2_score_reconstruction(model, array_curves),
        'holdout_r2_train': held_out[0] if held_out is not None else None,
        'holdout_r2': held_out[1] if held_out is not None else None,
        'cum_var_ratio': float(np.sum(model.explained_variance_ratio_)),
        # Rank-k PCA is fully described by its loadings and mean, and by
        # Eckart-Young it is the best possible linear reconstruction at this
        # rank - so it is the floor any nonlinear method has to beat.
        'n_params': int(model.components_.size + model.mean_.size),
        'detail': 'optimal linear rank-k',
    }
    return metrics, reconstructed


def autoencoder_metrics(array_curves, latent_dim):
    encoded, model = calculate_autoencoder(encoding_dim=latent_dim, array_curves=array_curves,
                                           epochs=AE_EPOCHS, log_dir=AE_LOG_DIR,
                                           val_fraction=VAL_FRACTION, seed=SPLIT_SEED)
    reconstructed = reconstruct_traces_autoencoder(encoded, model)

    data = torch.FloatTensor(array_curves)
    # Same fraction and seed as the training call, so this really is the slice
    # the weights were selected against and never trained on.
    train_data, val_data = split_train_val(data, VAL_FRACTION, SPLIT_SEED)

    kind = type(model).__name__
    metrics = {
        'method': f'Autoencoder ({"conv, nonlinear" if "Conv" in kind else "dense, nonlinear"})',
        'latent_dim': latent_dim,
        'n_samples': int(array_curves.shape[0]),
        'n_features': int(array_curves.shape[1]),
        'mse': mean_squared_error(model, data),
        'r2': explained_variance(model, data),
        # Scored on the training half, so it sits next to the held-out figure on
        # the same footing as PCA's: the gap between them is the overfitting.
        'holdout_r2_train': explained_variance(model, train_data),
        'holdout_r2': explained_variance(model, val_data) if val_data is not None else None,
        # Only meaningful for an orthogonal basis; the AE has no per-component
        # variance decomposition to report.
        'cum_var_ratio': None,
        'n_params': int(sum(p.numel() for p in model.parameters())),
        'detail': f'{AE_EPOCHS} epoch budget, early stopping',
    }
    return metrics, reconstructed


def shared_split_indices(n_samples, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    """The exact studies the autoencoder holds out, as indices.

    Both methods have to be scored on the SAME patients or the comparison is
    noise: pca_viewer.holdout_r2 splits with numpy's PCG64 while
    AutoencoderDm.split_train_val splits with torch.randperm, so the same seed
    produced two different held-out sets and the two R2 columns were not
    comparable.

    This mirrors split_train_val exactly and must stay in step with it; the
    autoencoder picks its own split internally from the same (fraction, seed),
    so reproducing it here is what ties the two together. Returns None when the
    cohort is too small to split, matching that function's own refusal.
    """
    n_val = int(round(n_samples * val_fraction))
    if val_fraction <= 0 or n_val < 1 or n_samples - n_val < 2:
        return None
    order = torch.randperm(n_samples, generator=torch.Generator(device='cpu').manual_seed(seed))
    return order[:n_val].numpy(), order[n_val:].numpy()


def baseline_metrics(array_curves):
    """The do-nothing reference: predict the cohort mean curve for everyone.

    R2 is defined against exactly this, so it scores 0 by construction. It is on
    the table because an R2 of 0.33 reads very differently once it is visible
    that 0.00 is the floor and 1.00 the ceiling - and because a method scoring
    below this row is worse than useless, which is otherwise easy to miss.
    """
    mean_curve = array_curves.mean(axis=0)
    return {
        'method': 'Mean curve (baseline)',
        'latent_dim': 0,
        'n_samples': int(array_curves.shape[0]),
        'n_features': int(array_curves.shape[1]),
        'mse': float(np.mean((array_curves - mean_curve) ** 2)),
        'r2': 0.0,
        'holdout_r2_train': 0.0,
        'holdout_r2': 0.0,
        'cum_var_ratio': None,
        'n_params': int(mean_curve.size),
        'detail': 'no reduction; reference floor',
    }


def average_metrics(per_segment):
    """Collapse one metrics dict per segment into a single row.

    Every segment carries the same studies, so a plain mean is the cohort-level
    figure; a segment whose split was refused drops out of that mean instead of
    poisoning it with a NaN.
    """
    averaged = dict(per_segment[0])
    for key in ['mse', 'r2', 'holdout_r2_train', 'holdout_r2', 'cum_var_ratio']:
        values = [m[key] for m in per_segment if m[key] is not None and not np.isnan(m[key])]
        averaged[key] = float(np.mean(values)) if values else None
    averaged['latent_dim'] = per_segment[0]['latent_dim']
    # Summed, not averaged: every segment carries its own model, so the cost of
    # the representation is all of them together.
    averaged['n_params'] = int(sum(m.get('n_params') or 0 for m in per_segment))
    averaged['detail'] = f"{len(per_segment)} segment models, mean of per-segment scores"
    return averaged


class ComparisonViewer:
    def __init__(self, pathname='/tab-Comparison'):
        self.pathname = pathname
        self.data_source = None
        self.table_name = None
        self.isImputed = None
        self.studyType = 'BL'
        self.df, self.tracesDF, self.study_nr = None, None, None
        self.study_nr_imputed, self.study_nr_original = None, None
        self.concat_tracesDF, self.array_curves = None, None
        self.pca_tracesDF, self.ae_tracesDF = None, None
        self.concat_traces_by, self.latent_dim = None, None
        self.metrics = []
        self.status = ''

    def load_study_nr_dropdown(self):
        if self.data_source != 'GE_Philips':
            trace_loader = LoadTraces(self.studyType, self.table_name, self.isImputed, isComponents=True)
            self.df, self.study_nr_imputed, self.study_nr_original = (
                trace_loader.tracesDF, trace_loader.study_nr_imputed, trace_loader.study_nr_original)
        else:
            trace_loader1 = LoadTraces(self.studyType, self.table_name[0], self.isImputed, isComponents=True)
            trace_loader2 = LoadTraces(self.studyType, self.table_name[1], self.isImputed, isComponents=True)
            df1, df2 = trace_loader1.tracesDF, trace_loader2.tracesDF
            if df1 is None or df2 is None:
                raise ValueError("Failed to load traces for GE_Philips data source")
            self.df = pd.concat([df1, df2], ignore_index=True)
            self.study_nr_imputed = list(set(trace_loader1.study_nr_imputed + trace_loader2.study_nr_imputed))
            self.study_nr_original = list(set(trace_loader1.study_nr_original + trace_loader2.study_nr_original))

        self.df = self.df[~self.df['curve_segment'].str.lower().str.contains('apex', na=False)]
        self.df['cardiac_view'] = self.df['cardiac_view'].str.lower()
        self.df['segment_wall'] = self.df['segment_wall'].str.lower().str.strip()
        dropna_tracesDF = self.df.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[~dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0))]
        self.tracesDF = dropna_tracesDF.groupby('study_nr').filter(lambda x: len(x) == 18)
        self.tracesDF = self.tracesDF.sort_values(by=['study_nr', 'id_position'])
        self.study_nr = np.sort(self.tracesDF['study_nr'].unique())

    def compare(self, concat_traces_by, latent_dim):
        self.concat_traces_by = concat_traces_by
        self.latent_dim = latent_dim
        self.metrics = []

        if concat_traces_by == 'full_traces':
            self.compare_full_traces()
        else:
            self.compare_segments()

    def compare_full_traces(self):
        concatenated_strain = CurveConcatenator(self.tracesDF, isReconstructed=False)
        self.concat_tracesDF = concatenated_strain.concatenate_traces_par()
        self.array_curves = np.array(self.concat_tracesDF['concatenated_trace'].tolist())

        # Both models see exactly the same matrix - that is the whole point of
        # comparing at equal latent dimension.
        pca_row, pca_reconstructed = pca_metrics(self.array_curves, self.latent_dim)
        ae_row, ae_reconstructed = autoencoder_metrics(self.array_curves, self.latent_dim)
        self.metrics = [baseline_metrics(self.array_curves), pca_row, ae_row]

        self.pca_tracesDF = self.split_concatenated(pca_reconstructed)
        self.ae_tracesDF = self.split_concatenated(ae_reconstructed)
        self.status = (f'{len(self.study_nr)} studies, {self.array_curves.shape[1]} features, '
                       f'latent dim {self.latent_dim} (PCA used {pca_row["latent_dim"]}).')

    def compare_segments(self):
        self.tracesDF['curve_segment'] = self.tracesDF['curve_segment'].str.strip()
        pca_frames, ae_frames, pca_rows, ae_rows, base_rows = [], [], [], [], []

        for curve_segment in self.tracesDF['curve_segment'].unique():
            segment_df = self.tracesDF[
                self.tracesDF['curve_segment'] == curve_segment].reset_index(drop=True)
            # literal_eval yields a tuple per row (traces are stored unbracketed),
            # and Series.round() raises TypeError on tuples - round the stacked
            # array instead. Same fix as prepare_segment_traces().
            array_curves = np.array(
                [list(t) for t in segment_df['strain_trace'].apply(ast.literal_eval)],
                dtype=float).round(6)

            pca_row, pca_reconstructed = pca_metrics(array_curves, self.latent_dim)
            ae_row, ae_reconstructed = autoencoder_metrics(array_curves, self.latent_dim)
            pca_rows.append(pca_row)
            ae_rows.append(ae_row)
            base_rows.append(baseline_metrics(array_curves))

            pca_frames.append(self.segment_frame(segment_df, curve_segment, pca_reconstructed))
            ae_frames.append(self.segment_frame(segment_df, curve_segment, ae_reconstructed))

        self.metrics = [average_metrics(base_rows), average_metrics(pca_rows),
                        average_metrics(ae_rows)]
        self.pca_tracesDF = pd.concat(pca_frames, ignore_index=True)
        self.ae_tracesDF = pd.concat(ae_frames, ignore_index=True)
        self.status = (f'{len(self.study_nr)} studies, one model per segment over '
                       f'{len(pca_rows)} segments; the table reports the mean across segments.')

    def split_concatenated(self, reconstructed_curves):
        """Cut each concatenated reconstruction back into its 18 segments."""
        reconstructed_curves = np.around(reconstructed_curves, decimals=6)
        traces = []
        for idx_rec, curve in enumerate(reconstructed_curves):
            start = 0
            study_nr = self.concat_tracesDF['study_nr'][idx_rec]
            for curve_len, curve_seg, id_seg in zip(self.concat_tracesDF['concat_curve_len'][idx_rec],
                                                    self.concat_tracesDF['concat_curve_segm'][idx_rec],
                                                    self.concat_tracesDF['concat_id_segment'][idx_rec]):
                traces.append({'study_nr': study_nr,
                               'curve_segment': curve_seg,
                               'id_position': id_seg,
                               'strain_trace': curve[start:start + int(curve_len)]})
                start += int(curve_len)
        return pd.DataFrame(traces)

    def segment_frame(self, segment_df, curve_segment, reconstructed_curves):
        reconstructed_curves = np.around(reconstructed_curves, decimals=6)
        traces = [{'study_nr': segment_df['study_nr'][idx_rec],
                   'curve_segment': curve_segment,
                   'id_position': int(segment_df['id_position'][idx_rec]),
                   'strain_trace': curve}
                  for idx_rec, curve in enumerate(reconstructed_curves)]
        return pd.DataFrame(traces)

    def get_content(self):
        return html.Div([
            # A fresh id per page load, so a second open tab cannot overwrite
            # this one's fitted models mid-plot.
            dcc.Store(id='cmp-session-id', data=new_session_id()),
            html.H3('PCA vs Autoencoder'),
            html.P('Both methods are fitted on the same feature matrix at the same latent '
                   'dimension, so the reconstruction errors are directly comparable.'),
            dbc.Row([
                html.Label('Select Data Source:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='cmp-data-source-dropdown',
                    options=[
                        {'label': 'Speqle', 'value': 'Speqle'},
                        {'label': 'GE', 'value': 'GE'},
                        {'label': 'Philips', 'value': 'Philips'},
                        {'label': 'GE + Philips', 'value': 'GE_Philips'}
                    ],
                    value='GE_Philips',
                    style={'width': '300px'}
                ),
                html.Br(),
                html.Label('Select Concatenation Type:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='cmp-concat-dropdown',
                    options=[
                        {'label': 'Full Traces', 'value': 'full_traces'},
                        {'label': 'Segment', 'value': 'segment'}
                    ],
                    value='full_traces',
                    style={'width': '400px', 'backgroundColor': 'white'}
                ),
            ]),
            html.Br(),
            html.P('Latent Dimension (PCA components = Autoencoder encoding dim):'),
            dcc.Slider(
                id='cmp-latent-slider',
                min=2, max=64, value=8, step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Br(),
            html.Button('Compare', id='cmp-run-button', style={'width': '200px'}, n_clicks=0),
            html.Br(),
            html.Div(id='cmp-status-txt', children='',
                     style={'fontStyle': 'italic', 'margin': '8px 0'}),
            html.Div(id='cmp-metrics-table', children=''),
            html.Br(),
            html.Label('Select Study Number:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(id='cmp-study-dropdown', style={'width': '200px'}),
            html.Div(id='cmp-error-txt', children='', style={'margin': '8px 0'}),
            dcc.Graph(id='cmp-graph'),
        ],
            style={'marginLeft': '20px'}
        )
