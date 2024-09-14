import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA


class PCAStrainTraces:
    def __init__(self, concatenated_tracesDF, studyType, concat_traces_by):
        self.reconstructed_tracesDF = pd.DataFrame()
        self.segment_tracesDF = None
        self.rec_traces_df = None
        self.PCA_tracesDF = pd.DataFrame()
        self.studyType = studyType
        self.concat_traces_by = concat_traces_by
        self.explained_variance, self.pca_result = None, None
        self.pca_95_var, self.array_curves = None, None
        self.concatenated_tracesDF = concatenated_tracesDF.reset_index(drop=True)

    def prepare_all_traces(self):
        curve = self.concatenated_tracesDF['concatenated_trace'].tolist()
        self.array_curves = np.array(curve)

    def prepare_segment_traces(self):
        unique_curve_segments = self.concatenated_tracesDF['curve_segment'].unique()
        unique_curve_segments = unique_curve_segments[unique_curve_segments != '0']
        for curve_segment in tqdm(unique_curve_segments):
            self.segment_tracesDF = self.concatenated_tracesDF[
                self.concatenated_tracesDF['curve_segment'] == curve_segment].reset_index(drop=True)
            segment_trace = self.segment_tracesDF['strain_trace'].apply(ast.literal_eval)
            segment_trace = segment_trace.round(6)
            self.array_curves = np.array(segment_trace.tolist())
            self.calculate_pca(curve_segment)
            self.reconstruct_traces(curve_segment)
            self.reconstructed_tracesDF = pd.concat([self.reconstructed_tracesDF, self.PCA_tracesDF], ignore_index=True)
            pass

    def calculate_pca(self, curve_segment):
        pca = PCA().fit(self.array_curves)
        cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
        print(f'Number of components that explain 95% of variance: {n_components_95}' + (
            f' in segment {curve_segment}' if self.concat_traces_by == 'segment' else ''))
        self.pca_95_var = PCA(n_components=n_components_95)
        self.pca_result = self.pca_95_var.fit_transform(self.array_curves)

    def reconstruct_traces(self, curve_segment):
        start = 0
        pca_traces = []
        reconstructed_curves = self.pca_95_var.inverse_transform(self.pca_result)
        rec_pca_curves = np.around(reconstructed_curves, decimals=6)
        for idx_rec, curve in enumerate(rec_pca_curves):
            if self.concat_traces_by == 'full_traces':
                study_nr = self.concatenated_tracesDF['study_nr'][idx_rec]
                concatenated_dic = self.concatenated_traces(start, idx_rec, curve, study_nr, pca_traces)
                self.PCA_tracesDF = pd.DataFrame(concatenated_dic)
                pass

            elif self.concat_traces_by == 'segment':
                study_nr = self.segment_tracesDF['study_nr'][idx_rec]
                concatenated_dic = self.segment_traces(idx_rec, curve, study_nr, pca_traces, curve_segment)
                self.PCA_tracesDF = pd.DataFrame(concatenated_dic)

    def concatenated_traces(self, start, idx_rec, curve, study_nr, pca_traces):
        original_curve_len = self.concatenated_tracesDF['concat_curve_len'][idx_rec]
        original_curve_seg = self.concatenated_tracesDF['concat_curve_segm'][idx_rec]
        original_id_seg = self.concatenated_tracesDF['concat_id_segment'][idx_rec]
        original_segment_wall = self.concatenated_tracesDF['concat_segment_wall'][idx_rec]
        original_cardiac_view = self.concatenated_tracesDF['concat_cardiac_view'][idx_rec]

        for curve_len, curve_seg, id_seg, seg_wall, cardiac_v in zip(original_curve_len, original_curve_seg,
                                                                     original_id_seg, original_segment_wall,
                                                                     original_cardiac_view):
            curve_segment = curve[start:start + int(curve_len)]
            pca_traces.append({'study_nr': study_nr,
                               'segment_wall': seg_wall,
                               'cardiac_view': cardiac_v,
                               'curve_segment': curve_seg,
                               'id_position': id_seg,
                               'strain_trace': curve_segment})
            start += int(curve_len)
        return pca_traces

    def segment_traces(self, idx_rec, curve, study_nr, pca_traces, curve_segment):
        pca_traces.append({'study_nr': study_nr,
                           'segment_wall': self.segment_tracesDF['segment_wall'][idx_rec],
                           'cardiac_view': self.segment_tracesDF['cardiac_view'][idx_rec],
                           'curve_segment': curve_segment,
                           'id_position': int(self.segment_tracesDF['id_position'][idx_rec]),
                           'strain_trace': curve})
        return pca_traces
