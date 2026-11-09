import ast
import json

from dash import dcc, html, Input, Output, dash, State, callback, ctx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler

from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertInterpTracesToDB
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbcon
from pattern_recognition.utils.plot_dics import segment_color_dict
from pattern_recognition.data.loadTraces import LoadTraces
from dash_viewer.dimensionality_reduction.AutoencoderDm import calculate_autoencoder, \
    reconstruct_traces_autoencoder
from dash_viewer.dimensionality_reduction.pca_viewer import (calculate_pca, holdout_r2,
                                                             reconstruct_traces_pca)
from dash_viewer.session_state import RunStore, new_session_id

# Run state lives here, keyed by the session id the page carries, so two open
# tabs cannot overwrite each other's fitted model mid-plot.
_runs = RunStore()

@callback(
    [Output('study_nr-dropdown_dr', 'options'),
     Output('study_nr-dropdown_dr', 'value'),
     Output('dm-variance-txt', 'children')],
    [Input('pca-button', 'n_clicks')],
    [State('data-source-dropdown_dr', 'value'),
     State('concat_traces_by-dropdown', 'value'),
     State('url', 'pathname'),
     State('id-slider-dm', 'value'),
     State('dm-session-id', 'data')]
)
def update_dropdown_and_load_study_nrs(pca_btn, data_source, concat_traces_by, pathname, n_components,
                                       session_id):
    dropdown_options = []
    if "pca-button" == ctx.triggered_id and pca_btn > 0:
        data_source_mapping = {
            'Speqle': 'speqle_segmentalStrainInterp_T',
            'GE': 'st_segmentalStrainInterp_T',
            'Philips': 'phi_segmentalStrainInterp_T',
            'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
        }
        if data_source in data_source_mapping:
            table_name = data_source_mapping[data_source]
        else:
            table_name = None
            raise ValueError("Invalid data source provided")

        run = DimensionalityRedViewer(pathname)
        run.data_source = data_source
        run.table_name = table_name
        run.load_study_nr_dropdown()
        run.trigger_callbacks(concat_traces_by, n_components)
        # Published only once the run is complete: a half-built run must never be
        # visible to the plotting callbacks.
        _runs.put(session_id, run)
        dropdown_options = [{'label': i, 'value': i} for i in run.study_nr]
        summary = [html.Div(line) for line in run.variance_summary]

        # The selection is cleared on every completed run: keeping the previous
        # study_nr leaves the plot callback untriggered, so the graphs would go on
        # showing the previous run's reconstruction under the new options.
        return [dropdown_options, None, summary]

    return dash.no_update, dash.no_update, dash.no_update



@callback(
    [Output('pca-graph', 'figure'),
     Output('origin_trace', 'figure')],
    [Input('study_nr-dropdown_dr', 'value'),
    State('url', 'pathname'),
    State('dm-session-id', 'data')]
)
def plot_traces_dm(study_nr, data_type, session_id):
    instance_dm = _runs.get(session_id)
    if study_nr is not None and instance_dm is not None:
        data_type = data_type.split('-')[-1]
        if data_type == 'PCA':
            n_components = instance_dm.n_components_used
        else:
            n_components = instance_dm.n_components
        reconstructed_df = instance_dm.DM_tracesDF[instance_dm.DM_tracesDF['study_nr'] ==
                                                   study_nr].sort_values(by='id_position')

        original_df = instance_dm.tracesDF[instance_dm.tracesDF['study_nr'] ==
                                           study_nr].sort_values(by='id_position')

        fig_original = plot_curve(original_df, study_nr, n_components, 'Interpolated',
                                  instance_dm.pathname)
        fig_reconstructed = plot_curve(reconstructed_df, study_nr, n_components, 'Reconstructed',
                                       instance_dm.pathname)

        return fig_reconstructed, fig_original
    else:
        return go.Figure(), go.Figure()


@callback(
    Output('id-data-txt', 'children'),
    Input('savetoDB-button', 'n_clicks'),
    [State('concat_traces_by-dropdown', 'value'),
     State('url', 'pathname'),
     State('dm-session-id', 'data')]
)
def handle_db_click(insert_btn, concat_traces_by, data_type, session_id):

    if "savetoDB-button" == ctx.triggered_id and insert_btn > 0:
        instance_dm = _runs.get(session_id)
        if instance_dm is None:
            return 'Nothing to save yet - run the reduction first.'
        print('Saving to DB')
        # The run's own concatenation type, not the live dropdown: the user is free
        # to change the dropdown after a run, and that must not relabel data the
        # previous run produced.
        concat_traces_by = instance_dm.concat_traces_by or concat_traces_by
        pathname = 1
        pca_result_df = pd.DataFrame()
        data_type = data_type.split('-')[-1]
        recons_traces_df = instance_dm.DM_tracesDF.copy()
        data_source = instance_dm.data_source
        if data_type == 'PCA':
            n_components = instance_dm.n_components_used
        else:
            n_components = instance_dm.n_components

        # Segment mode fits per segment, and on the "Optimal" setting each one
        # can legitimately pick a different k (measured: 4 for midAnt/midLat, 5
        # elsewhere). n_components_used holds only the last segment's, so the key
        # would claim one k for all 18.
        #
        # The key string carries every distinct k, while the n_components column
        # stays numeric - it is typed float in the table, so a '4_5' string would
        # fail the insert. With a uniform k both are just that k, leaving the
        # common fixed-k case byte-identical to before.
        components_key = n_components
        if instance_dm.segment_k:
            ks = sorted({int(k) for k in instance_dm.segment_k.values()})
            n_components = max(ks)
            components_key = ks[0] if len(ks) == 1 else '_'.join(str(k) for k in ks)

        recons_traces_df['strain_trace'] = recons_traces_df['strain_trace'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        # Sets, not lists: this runs once per row over the whole cohort, and `in`
        # on a list made it quadratic.
        imputed_ids = set(instance_dm.study_nr_imputed or [])
        original_ids = set(instance_dm.study_nr_original or [])
        recons_traces_df['isImputed'] = recons_traces_df['study_nr'].apply(
            lambda x: 1 if x in imputed_ids else (0 if x in original_ids else None)
        )
        unknown = int(recons_traces_df['isImputed'].isna().sum())
        if unknown:
            # These used to fall through the reconstructed_type ternary below and
            # be written as originals, silently mislabelling their provenance.
            print(f"[WARN] {unknown} row(s) match neither the imputed nor the original "
                  f"study list; they are being saved as isImputed=0.")
        recons_traces_df['study_type'] = 'BL'
        recons_traces_df['reconstructed_type'] = recons_traces_df.apply(
            lambda
                row: f"{data_type}-{concat_traces_by}-{components_key}-{1 if row['isImputed'] == 1 else 0}",
            axis=1
        )

        recons_traces_df['n_components'] = n_components
        recons_traces_df['data_source'] = data_source
        recons_traces_df = recons_traces_df.reset_index(drop=True)
        new_column_order = [
            'study_nr', 'curve_segment', 'id_position', 'strain_trace', 'study_type',
            'reconstructed_type', 'n_components', 'isImputed', 'data_source'
        ]
        recons_traces_df = recons_traces_df[new_column_order]
        replace_existing_rows('calculated_reconstructedTraces_T', recons_traces_df,
                              ('reconstructed_type', 'data_source'))
        InsertInterpTracesToDB(recons_traces_df, 'calculated_reconstructedTraces_T', True)

        unique_study_nr = instance_dm.DM_tracesDF['study_nr'].drop_duplicates()

        if data_type == 'PCA':
            recons_traces_df = recons_traces_df.drop_duplicates(subset='study_nr', keep='first')
            mapping_series_recon = recons_traces_df.set_index('study_nr')['reconstructed_type']
            mapping_series_isImp = recons_traces_df.set_index('study_nr')['isImputed']

            # In segment mode a study's representation is every segment's codes
            # concatenated, not the last segment's. stitch_segment_scores()
            # returns None in full-trace mode, where pca_result already is the
            # whole-study representation.
            stitched = instance_dm.stitch_segment_scores()
            if stitched is not None:
                study_ids, scores, _ = stitched
            else:
                study_ids, scores = unique_study_nr.values, instance_dm.pca_result

            data = []
            for study_nr_i, row in zip(study_ids, scores):
                row_json = json.dumps(np.asarray(row).tolist())
                data.append({
                    'study_nr': study_nr_i,
                    'pca_results': row_json,
                    'reconstructed_type': mapping_series_recon.get(study_nr_i, 'Unknown'),
                    'n_components': n_components,
                    'isImputed': mapping_series_isImp.get(study_nr_i, False),
                    'data_source': data_source
                })

            pca_result_df = pd.DataFrame(data)
            replace_existing_rows('calculated_pca_results_T', pca_result_df,
                                  ('reconstructed_type', 'data_source'))
            InsertInterpTracesToDB(pca_result_df, 'calculated_pca_results_T', False)

            def model_row(pca_model, label):
                return {
                    'components_': json.dumps(pca_model.components_.tolist()),
                    'explained_variance_': json.dumps(pca_model.explained_variance_.tolist()),
                    'explained_variance_ratio_': json.dumps(
                        pca_model.explained_variance_ratio_.tolist()),
                    # Variance % per PC
                    'mean_': json.dumps(pca_model.mean_.tolist()),
                    'n_components_': pca_model.n_components_,
                    'n_features_': pca_model.n_features_in_,
                    'singular_values_': json.dumps(pca_model.singular_values_.tolist()),
                    # Singular values (importance of PCs)
                    'reconstructed_type': label,
                    'n_components': n_components,
                    'isImputed': 0.1,
                    'data_source': data_source
                }

            # Segment mode fits one basis per segment, so every basis is stored -
            # keyed by segment, since a single row could only ever describe one of
            # them and reconstruction needs all 18.
            if instance_dm.segment_models:
                pca_var_df = pd.DataFrame(
                    [model_row(model, f'{concat_traces_by}-{segment}')
                     for segment, model in sorted(instance_dm.segment_models.items())])
            else:
                pca_var_df = pd.DataFrame([model_row(instance_dm.pca_var_comp, concat_traces_by)])
            # This table keys on the concatenation type alone, so n_components has
            # to join the key or a 3-component run would wipe a 59-component one.
            replace_existing_rows('calculated_pca_params_T', pca_var_df,
                                  ('reconstructed_type', 'data_source', 'n_components'))
            InsertInterpTracesToDB(pca_var_df, 'calculated_pca_params_T', True)

        elif data_type == 'AutoencoderDm':
            recons_traces_df = recons_traces_df.drop_duplicates(subset='study_nr', keep='first')
            mapping_series_recon = recons_traces_df.set_index('study_nr')['reconstructed_type']
            mapping_series_isImp = recons_traces_df.set_index('study_nr')['isImputed']

            # Same stitching as the PCA branch: in segment mode one encoder is
            # trained per segment, so a study's code is all 18 concatenated, not
            # whichever segment trained last.
            stitched = instance_dm.stitch_segment_scores()
            if stitched is not None:
                study_ids, codes, _ = stitched
            else:
                study_ids, codes = unique_study_nr.values, instance_dm.encoded_data

            data = []
            for study_nr_i, row in zip(study_ids, codes):
                row_json = json.dumps(np.asarray(row).tolist())
                data.append({
                    'study_nr': study_nr_i,
                    'encoded_vector': row_json,
                    'reconstructed_type': mapping_series_recon.get(study_nr_i, 'Unknown'),
                    'encoding': n_components,
                    'isImputed': mapping_series_isImp.get(study_nr_i, False),
                    'data_source': data_source
                })

            autoencoder_result_df = pd.DataFrame(data)
            replace_existing_rows('calculated_dr_encoded_data_T', autoencoder_result_df,
                                  ('reconstructed_type', 'data_source'))
            InsertInterpTracesToDB(autoencoder_result_df, 'calculated_dr_encoded_data_T', True)

        print('Saving completed....')
        txt_save = f' Inserting to DB completed'
        return txt_save
    return dash.no_update


def replace_existing_rows(table_key, frame, key_columns):
    """Delete the rows this save is about to rewrite.

    InsertInterpTracesToDB only ever inserts, so saving the same configuration
    twice left two identical copies in the table. Nothing downstream notices:
    load_data_traces() filters on reconstructed_type, so both copies load and
    every study silently arrives with doubled rows - which then looks like a
    duplicated exam to the classifier's fold grouping.

    Only the exact key combinations present in `frame` are removed, so other
    configurations sharing the table are left alone. Values are passed as query
    parameters rather than interpolated: delete_query_where() builds
    "WHERE col = {value}" unquoted, which is invalid SQL for a string key like
    'PCA-segment-3-0'.
    """
    combos = frame[list(key_columns)].drop_duplicates().values.tolist()
    if not combos:
        return 0

    where = ' AND '.join(f'{col} = ?' for col in key_columns)
    deleted = 0
    dbConParams = dbConParameters
    db = dbcon(driver=dbConParams['driver'], server=dbConParams['server'],
               database=dbConParams['database'])
    try:
        db.connect()
        for values in combos:
            db.cursor.execute(
                f"DELETE FROM {dbTableNames[table_key]} WHERE {where}",
                *[v.item() if hasattr(v, 'item') else v for v in values])
            deleted += db.cursor.rowcount if db.cursor.rowcount > 0 else 0
        db.cursor.commit()
    except Exception as e:
        print(f"[WARN] could not clear previous rows in {table_key} ({e}); "
              f"the insert below may duplicate them.")
    finally:
        db.close_connection()

    if deleted:
        print(f"[INFO] replaced {deleted} existing row(s) in {table_key}.")
    return deleted


def plot_curve(trace, study_nr, n_components, traceType, pathname):
    fig = go.Figure()
    title_str = ''
    trace = trace.reset_index(drop=True)
    for idx_curve, curve in enumerate(trace['strain_trace']):
        segment_name = trace['curve_segment'][idx_curve]
        segment_name = segment_name.strip()
        if traceType == 'Interpolated':
            curve = ast.literal_eval(curve)
            curve = list(map(float, curve))
            title_str = f'{traceType} Strain Curve - Study {study_nr}'
        else:
            if pathname == '/tab-PCA':
                title_str = (f'{traceType} Strain Curve -\n'
                             f'Study {study_nr} using {n_components} components of the explained variance')
            elif pathname == '/tab-AutoencoderDm':
                title_str = (f'{traceType} Strain Curve -\n'
                             f'Study {study_nr} using {n_components} latent (encoding) dimensions')

        fig.add_trace(go.Scatter(y=curve, mode='lines', name=segment_name,
                                 line=dict(color=segment_color_dict[segment_name]), ))

    fig.update_layout(title=title_str, title_x=0.5,
                      autosize=False, height=600)
    return fig


def load_table_db(view_name, column_name, operator, value):
    try:
        dbConParams = dbConParameters
        db = dbcon(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.execute_query(sql_queries.select_all_data_where(view_name, column_name, operator, value))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# def process_strain_traces(combined_data):
#     def convert_to_array(x):
#         if '[' in x:
#             return np.array([float(i) for i in x.strip('[]').split(',')])
#         else:
#             return np.array([float(i) for i in x.split(',')])
#
#     strain_traces = combined_data['strain_trace'].apply(convert_to_array)
#     flattened_strain_traces = np.concatenate(strain_traces.values)
#     scaler = StandardScaler()
#     scaler.fit(flattened_strain_traces.reshape(-1, 1))
#     standardized_strain_traces = strain_traces.apply(
#         lambda x: scaler.transform(x.reshape(-1, 1)).flatten()
#     )
#
#     return standardized_strain_traces


class DimensionalityRedViewer:
    def __init__(self, pathname):
        self.n_components_used = None
        self.trained_model = None
        self.encoded_data = None
        self.study_nr_imputed = None
        self.study_nr_original = None
        self.mean_model = None
        self.concat_echo_params = None
        self.df = None
        self.data_source = None
        self.table_name = None
        self.isImputed = None
        self.data_type = None
        self.pca_model = None
        self.studyType = 'BL'
        self.n_components = None
        self.pathname = pathname
        self.array_curves, self.pca_result, self.pca_var_comp, self.tracesDF = None, None, None, None
        self.explained_var_str, self.PCA_rec_tracesDF, self.reconstructed_tracesDF = None, None, None
        self.calculate_PCA, self.DM_tracesDF, self.array_curves, self.study_nr = None, None, None, None
        self.concat_traces_by, self.segment_tracesDF, self.concat_tracesDF, self.content = None, None, None, None
        self.trace_scaler, self.n_trace_features = None, None
        self.variance_summary = []
        # Per-segment latent codes, models and component counts, keyed by segment.
        # Empty in full-trace mode, where one model covers the whole curve.
        self.segment_scores, self.segment_models = {}, {}
        self.segment_k, self.segment_study_order = {}, {}

    def load_echo_params(self):
        unique_study_nr_list = self.study_nr.tolist()
        unique_study_nr_str = ', '.join(map(str, unique_study_nr_list))
        echo_params_DF = load_table_db('echo_biomarkers_view', 'study_nr', 'in', unique_study_nr_str)
        echo_paramsDF = echo_params_DF[~echo_params_DF['View'].str.contains('RV')]
        echo_paramsDF_ = echo_paramsDF.dropna(axis=1, how='all')
        study_nr = echo_paramsDF['study_nr']
        df_to_concat = echo_paramsDF.drop(['study_nr', 'View', 'Segment'], axis=1)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_to_concat)
        data_scaled_df = pd.DataFrame(data_scaled, columns=df_to_concat.columns)
        data_scaled_df = pd.concat([study_nr, data_scaled_df], axis=1)
        self.concat_echo_params = data_scaled_df
        self.concat_echo_params['study_nr'] = self.concat_echo_params['study_nr'].astype(int)

        # echo_paramsDF['concatenated'] = data_scaled_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        # self.concat_echo_params = echo_paramsDF[['study_nr', 'concatenated']]
        pass

    def build_echo_matrix(self, study_nr_order):
        """Echo biomarkers as a numeric matrix, one row per study in study_nr_order.

        load_echo_params standardises the biomarkers but leaves one row per view,
        so collapse to a single row per study before aligning.
        """
        if self.concat_echo_params is None:
            self.load_echo_params()

        echo_df = self.concat_echo_params.groupby('study_nr').mean(numeric_only=True)
        echo_df = echo_df.reindex([int(nr) for nr in study_nr_order])
        # A study with no echo row - or one missing a single biomarker - sits at the
        # cohort mean, which is 0 post-standardisation. Dropping it instead would
        # break alignment with the trace rows.
        return echo_df.fillna(0.0).to_numpy(dtype=float)

    def combine_traces_with_echo(self, array_curves, study_nr_order):
        """Stack standardised traces and echo biomarkers into one feature matrix.

        The two blocks are in different units, so the traces are standardised per
        sample column to match the already-standardised echo params. Traces still
        carry more total variance because there are far more of them; that is
        deliberate - the echo params inform the components without dominating.
        """
        traces = np.asarray(array_curves, dtype=float)
        self.trace_scaler = StandardScaler().fit(traces)
        self.n_trace_features = traces.shape[1]
        echo_matrix = self.build_echo_matrix(study_nr_order)

        return np.hstack([self.trace_scaler.transform(traces), echo_matrix])

    def describe_pca(self, array_curves, label=None):
        """One line of 'what did these components buy me', for the tab.

        Reports held-out R2 next to the in-sample figure: explained_variance_ratio_
        is measured on the data the basis was fitted to, so on its own it says
        nothing about how the basis handles a study it has not seen.
        """
        cumulative = float(np.sum(self.pca_var_comp.explained_variance_ratio_))
        prefix = f'{label}: ' if label else ''
        line = (f'{prefix}{self.n_components_used} components explain '
                f'{cumulative * 100:.2f}% of the variance (in-sample)')

        scores = holdout_r2(array_curves, self.n_components_used)
        if scores is not None:
            train_r2, held_out_r2, _ = scores
            line += f'; R2 {train_r2:.3f} train / {held_out_r2:.3f} held-out'
        self.variance_summary.append(line)
        print(line)

    def describe_autoencoder(self, array_curves, label=None):
        """The autoencoder's counterpart to describe_pca, so both tabs say something.

        calculate_autoencoder only prints its metrics, and the held-out split it
        selected on is internal to it - so the same split (same seed, same
        fraction) is rebuilt here to score the fitted model on data it was not
        trained on, next to the in-sample figure.
        """
        # torch is imported lazily: the module must stay importable on a machine
        # without it, and every other path through this viewer is PCA.
        import torch
        from dash_viewer.dimensionality_reduction.AutoencoderDm import (explained_variance,
                                                                        mean_squared_error,
                                                                        split_train_val)

        data = torch.FloatTensor(np.asarray(array_curves, dtype=float))
        train_data, val_data = split_train_val(data, val_fraction=0.15, seed=42)

        prefix = f'{label}: ' if label else ''
        line = (f'{prefix}{self.n_components} latent dims, '
                f'R2 {explained_variance(self.trained_model, train_data):.3f} train')

        if val_data is not None:
            line += (f' / {explained_variance(self.trained_model, val_data):.3f} held-out'
                     f' (MSE {mean_squared_error(self.trained_model, train_data):.4f} / '
                     f'{mean_squared_error(self.trained_model, val_data):.4f})')
        else:
            line += (f' (MSE {mean_squared_error(self.trained_model, train_data):.4f}; '
                     f'cohort too small to hold out)')

        self.variance_summary.append(line)
        print(line)

    def load_study_nr_dropdown(self):
        if self.data_source != 'GE_Philips':
            trace_loader = LoadTraces(self.studyType, self.table_name, self.isImputed, isComponents=True)
            self.df, self.study_nr_imputed, self.study_nr_original = (
                trace_loader.tracesDF, trace_loader.study_nr_imputed, trace_loader.study_nr_original)
        else:
            trace_loader1 = LoadTraces(self.studyType, self.table_name[0], self.isImputed, isComponents=True)
            trace_loader2 = LoadTraces(self.studyType, self.table_name[1], self.isImputed, isComponents=True)
            df1, study_nr_imputed1, study_nr_original1 = trace_loader1.tracesDF, trace_loader1.study_nr_imputed, trace_loader1.study_nr_original
            df2, study_nr_imputed2, study_nr_original2 = trace_loader2.tracesDF, trace_loader2.study_nr_imputed, trace_loader2.study_nr_original

            if df1 is not None and df2 is not None:
                self.df = pd.concat([df1, df2], ignore_index=True)
                self.study_nr_imputed = list(set(study_nr_imputed1 + study_nr_imputed2))
                self.study_nr_original = list(set(study_nr_original1 + study_nr_original2))
            else:
                raise ValueError("Failed to load traces for GE_Philips data source")

        self.df = self.df[~self.df['curve_segment'].str.lower().str.contains('apex', na=False)]
        self.df['cardiac_view'] = self.df['cardiac_view'].str.lower()
        self.df['segment_wall'] = self.df['segment_wall'].str.lower().str.strip() # .str.lower() changed on 07/08/2025
        dropna_tracesDF = self.df.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[~dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0))]
        self.tracesDF = dropna_tracesDF.groupby('study_nr').filter(lambda x: len(x) == 18)
        self.tracesDF = self.tracesDF.sort_values(by=['study_nr', 'id_position'])
        self.study_nr = np.sort(self.tracesDF['study_nr'].unique())

    def stitch_segment_scores(self):
        """One latent vector per study, made of every segment's codes side by side.

        In segment mode each segment has its own basis, so a study's
        representation is the 18 codes concatenated - 18 segments x k - not any
        single segment's. Saving self.pca_result instead stored whichever
        segment ran last and labelled it as the whole study.

        Segments are ordered by name so the vector's layout is stable between
        runs, and each segment's rows are reindexed onto a common study order:
        the per-segment frames are built by filtering the same sorted table, so
        they agree today, but a missing segment row would otherwise shift a
        study's codes into another study's slot.

        Returns (study_nr array, codes array of shape (n_studies, sum(k)),
        column labels) or None when there is nothing stitched.
        """
        if not self.segment_scores:
            return None

        segments = sorted(self.segment_scores)
        studies = sorted(set(self.study_nr.tolist()))
        blocks, labels = [], []

        for segment in segments:
            scores = np.asarray(self.segment_scores[segment], dtype=float)
            order = self.segment_study_order.get(segment, [])
            position = {study: i for i, study in enumerate(order)}
            k = scores.shape[1]

            block = np.full((len(studies), k), np.nan)
            for row, study in enumerate(studies):
                if study in position:
                    block[row] = scores[position[study]]
            blocks.append(block)
            labels += [f'{segment}_PC{i + 1}' for i in range(k)]

        return np.asarray(studies), np.hstack(blocks), labels

    def load_pca_data(self):
        self.trace_scaler, self.n_trace_features = None, None
        self.variance_summary = []
        if self.concat_traces_by == 'full_traces' or self.concat_traces_by == 'full_traces_echo':
            self.prepare_all_traces()
        elif self.concat_traces_by == 'segment' or self.concat_traces_by == 'segment_echo':
            self.prepare_segment_traces()

    def prepare_all_traces(self):
        concatenated_strain = CurveConcatenator(self.tracesDF, isReconstructed=False)
        self.concat_tracesDF = concatenated_strain.concatenate_traces_par()
        curve = self.concat_tracesDF['concatenated_trace'].tolist()
        self.array_curves = np.array(curve)

    def prepare_segment_traces(self):
        self.DM_tracesDF = pd.DataFrame()
        self.tracesDF['curve_segment'] = self.tracesDF['curve_segment'].str.strip()
        unique_curve_segments = (self.tracesDF['curve_segment'].unique())

        # One model per segment, so the per-segment results have to be collected
        # as they are produced. self.pca_result and friends are overwritten on
        # every iteration; reading them after the loop yields whichever segment
        # happened to run last, which is how the saved latent codes came to hold
        # a single segment labelled as the whole study.
        self.segment_scores, self.segment_models, self.segment_k = {}, {}, {}

        for curve_segment in unique_curve_segments:
            self.segment_tracesDF = self.tracesDF[
                self.tracesDF['curve_segment'] == curve_segment].reset_index(drop=True)

            # Traces are stored unbracketed ("0.0, 0.549, ..."), so literal_eval
            # returns a tuple per row. Series.round() then raises TypeError on
            # those tuples - rounding has to happen on the stacked array, not on
            # a Series of sequences. This is why segment mode never ran.
            self.array_curves = np.array(
                [list(t) for t in self.segment_tracesDF['strain_trace'].apply(ast.literal_eval)],
                dtype=float).round(6)

            if self.pathname == '/tab-PCA':
                if self.concat_traces_by == 'segment_echo':
                    self.load_echo_params()
                    segment_curves = self.combine_traces_with_echo(
                        self.array_curves, self.segment_tracesDF['study_nr'].tolist())
                    self.pca_result, self.pca_var_comp, self.n_components_used = calculate_pca(
                        self.n_components, segment_curves)
                    self.describe_pca(segment_curves, f'{curve_segment} + echo params')
                    self.mean_model = np.mean(self.array_curves, axis=0)
                else:
                    self.trace_scaler, self.n_trace_features = None, None
                    self.pca_result, self.pca_var_comp,self.n_components_used = calculate_pca(self.n_components, self.array_curves)
                    self.describe_pca(self.array_curves, curve_segment)
                    self.mean_model = np.mean(self.array_curves, axis=0)

                self.segment_scores[curve_segment] = self.pca_result
                self.segment_models[curve_segment] = self.pca_var_comp
                self.segment_k[curve_segment] = self.n_components_used

            elif self.pathname == '/tab-AutoencoderDm':
                log_dir = "runs/autoencoder_experiment"
                self.encoded_data, self.trained_model = calculate_autoencoder(encoding_dim=self.n_components,
                                                                              array_curves=self.array_curves,
                                                                              epochs=1000,
                                                                              log_dir=log_dir)
                self.describe_autoencoder(self.array_curves, curve_segment)
                self.segment_scores[curve_segment] = self.encoded_data
                self.segment_k[curve_segment] = self.n_components

            # The study order a segment's score rows are in, so the codes can be
            # stitched back onto the right studies after the loop.
            self.segment_study_order[curve_segment] = self.segment_tracesDF['study_nr'].tolist()

            self.reconstruct_traces(curve_segment)
            self.DM_tracesDF = pd.concat([self.DM_tracesDF, self.PCA_rec_tracesDF], ignore_index=True)
        self.DM_tracesDF = self.DM_tracesDF.sort_values(by='study_nr').reset_index()
        self.DM_tracesDF = self.DM_tracesDF.drop(columns=['index'])
        pass

    def reconstruct_traces(self, curve_segment=None):
        start = 0
        pca_traces = []
        if self.pathname == '/tab-PCA':
            reconstructed_curves = reconstruct_traces_pca(self.pca_result, self.pca_var_comp)
        elif self.pathname == '/tab-AutoencoderDm':
            reconstructed_curves = reconstruct_traces_autoencoder(self.encoded_data, self.trained_model)

        if self.trace_scaler is not None:
            # The echo params were extra columns on the right of the feature matrix:
            # drop them and return the traces to strain units before slicing.
            reconstructed_curves = self.trace_scaler.inverse_transform(
                reconstructed_curves[:, :self.n_trace_features])

        rec_pca_curves = np.around(reconstructed_curves, decimals=6)

        for idx_rec, curve in enumerate(rec_pca_curves):
            if self.concat_traces_by == 'full_traces' or self.concat_traces_by == 'full_traces_echo':
                study_nr = self.study_nr[idx_rec]
                self.concatenated_traces(start, idx_rec, curve, study_nr, pca_traces)

            elif self.concat_traces_by == 'segment' or self.concat_traces_by == 'segment_echo':
                study_nr = self.segment_tracesDF['study_nr'][idx_rec]
                self.segment_traces(idx_rec, curve, study_nr, pca_traces, curve_segment)

        # Built once from the finished list. Rebuilding the frame inside the loop
        # re-copied every row already appended, once per row.
        if self.concat_traces_by in ('segment', 'segment_echo'):
            self.PCA_rec_tracesDF = pd.DataFrame(pca_traces)
        else:
            self.DM_tracesDF = pd.DataFrame(pca_traces)

    def concatenated_traces(self, start, idx_rec, curve, study_nr, pca_traces):
        original_curve_len = self.concat_tracesDF['concat_curve_len'][idx_rec]
        original_curve_seg = self.concat_tracesDF['concat_curve_segm'][idx_rec]
        original_id_seg = self.concat_tracesDF['concat_id_segment'][idx_rec]

        for curve_len, curve_seg, id_seg in zip(original_curve_len, original_curve_seg, original_id_seg):
            curve_segment = curve[start:start + int(curve_len)]
            pca_traces.append({'study_nr': study_nr,
                               'curve_segment': curve_seg,
                               'id_position': id_seg,
                               'strain_trace': curve_segment})
            start += int(curve_len)

    def segment_traces(self, idx_rec, curve, study_nr, pca_traces, curve_segment):
        pca_traces.append({'study_nr': study_nr,
                           'curve_segment': curve_segment,
                           'id_position': int(self.segment_tracesDF['id_position'][idx_rec]),
                           'strain_trace': curve})
        return pca_traces

    def get_content(self):
        content = html.Div([
            # A fresh id per page load: each open tab keeps its own run state on
            # the server, so a second run cannot overwrite the first one's plots.
            dcc.Store(id='dm-session-id', data=new_session_id()),
            html.H3('Dimensionality Reduction'),
            dbc.Row([
                html.Br(),
                dcc.Dropdown(
                    id='data-source-dropdown_dr',
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
                html.Br(),

                html.Label('Select Concatenation Type:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='concat_traces_by-dropdown',
                    options=[
                        {'label': 'Full Traces', 'value': 'full_traces'},
                        {'label': 'Segment', 'value': 'segment'},
                        # The echo variants fit on traces plus the standardised echo
                        # biomarkers; both paths exist downstream, so offer them here.
                        {'label': 'Full Traces + Echo Params', 'value': 'full_traces_echo'},
                        {'label': 'Segment + Echo Params', 'value': 'segment_echo'}
                    ],
                    value='full_traces',
                    style={'width': '400px', 'backgroundColor': 'white'}
                ),
                html.Br(),

            ]),
            html.Br(),
            html.Br(),
            html.Button(
                'Perform DR', id='pca-button', style={'width': '200px'}, n_clicks=0),
            html.Br(),
            # What the chosen components actually bought, filled in by the run.
            html.Div(id='dm-variance-txt', children='',
                     style={'fontStyle': 'italic', 'marginBottom': '8px'}),
            html.Label('Select Study Number:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='study_nr-dropdown_dr',
                style={'width': '200px'}
            ),
            html.Br(),

            dbc.Row([
                dbc.Col(dcc.Graph(id='origin_trace'), width=6),
                dbc.Col(dcc.Graph(id='pca-graph'), width=6),
            ]), html.Br(),

            html.Button(
                'Save', id='savetoDB-button', style={'width': '200px', 'marginLeft': '10px'}, n_clicks=0),
            html.Br(),
            html.Div(id='id-data-txt', children=' ', style={'marginLeft': '10px'})

        ],
            style={'marginLeft': '20px'}
        )
        if self.pathname == '/tab-PCA':
            content.children.insert(3, html.P("Number of Components:"))
            content.children.insert(4, dcc.Slider(
                id='id-slider-dm',
                min=0,
                max=100,
                value=3,
                step=1,
                marks={
                    0: {'label': 'Optimal', 'style': {'color': '#777', 'font-weight': 'bold'}},
                    100: {'label': 'Maximum', 'style': {'color': '#0074D9', 'font-weight': 'bold'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ))

            # content.children.insert(4, dcc.Slider(
            #     id='id-slider-dm',
            #     min=0, max=60, value=59,
            #     step=1
            #     # min=0, max=1, value=0.95,
            #     # step=0.05
            # ))
        elif self.pathname == '/tab-AutoencoderDm':
            content.children.insert(3, html.P("Encoding Dimensions:"))
            content.children.insert(4, dcc.Slider(
                id='id-slider-dm',
                min=2, max=64, value=64,
                step=2
            ))

        return content

    def trigger_callbacks(self, concat_traces_by, n_components):
        self.concat_traces_by = concat_traces_by
        self.n_components = n_components
        self.load_pca_data()

        if concat_traces_by == 'full_traces' or concat_traces_by == 'full_traces_echo':
            if self.pathname == '/tab-PCA':
                self.data_type = 'PCA'

                if concat_traces_by == 'full_traces_echo':
                    self.load_echo_params()
                    array_curves = self.combine_traces_with_echo(
                        self.array_curves, self.concat_tracesDF['study_nr'].tolist())

                    self.pca_result, self.pca_var_comp, self.n_components_used = calculate_pca(
                        self.n_components, array_curves)
                    self.describe_pca(array_curves, 'traces + echo params')
                    self.mean_model = np.mean(self.array_curves, axis=0)
                else:
                    self.pca_result, self.pca_var_comp,self.n_components_used = calculate_pca(self.n_components, self.array_curves)
                    self.describe_pca(self.array_curves)
                    self.mean_model = np.mean(self.array_curves, axis=0)

            elif self.pathname == '/tab-AutoencoderDm':
                self.data_type = 'AutoencoderDm'
                log_dir = "runs/autoencoder_experiment"
                self.encoded_data, self.trained_model = calculate_autoencoder(encoding_dim=self.n_components,
                                                                              array_curves=self.array_curves,
                                                                              epochs=1000,
                                                                              log_dir=log_dir)
                self.describe_autoencoder(self.array_curves)

            # The full-trace path splits by the concatenated segment lengths, so it
            # has no single curve_segment to name.
            self.reconstruct_traces(None)
