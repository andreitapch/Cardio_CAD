import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures


def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val

def strip_and_get(series):
    return series.astype(str).str.strip().values


class CurveConcatenator:
    def __init__(self, segmental_tracesDF, isReconstructed):
        self.segmental_tracesDF = segmental_tracesDF
        self.isReconstructed = isReconstructed

    def concatenate_traces(self):
        print('Concatenating traces...')
        unique_study_nr = self.segmental_tracesDF['study_nr'].unique()
        concat_traces = []

        for study_nr in unique_study_nr:
            studyDf = self.segmental_tracesDF[self.segmental_tracesDF['study_nr'] == study_nr]
            tracesDF = pd.DataFrame(studyDf['strain_trace'].apply(safe_eval).tolist())
            concatenated_trace = np.concatenate(tracesDF.round(6).values)

            concat_curve_segm = strip_and_get(studyDf['curve_segment'])

            # concat_curve_segm = studyDf['curve_segment'].str.rstrip().values

            if not self.isReconstructed:
                concat_curve_len = studyDf['curve_length'].values
                concat_curve_segm_wall = strip_and_get(studyDf['segment_wall'])
                concat_curve_cardiac_view = strip_and_get(studyDf['cardiac_view'])
                concat_id_segment = strip_and_get(studyDf['id_position'])

                # concat_curve_segm_wall = studyDf['segment_wall'].str.rstrip().values
                # concat_curve_cardiac_view = studyDf['cardiac_view'].str.rstrip().values
                # concat_id_segment = studyDf['id_position'].str.rstrip().values

                concat_traces.append({
                    'study_nr': study_nr,
                    'concatenated_trace': concatenated_trace,
                    'concat_curve_len': concat_curve_len,
                    'concat_id_segment': concat_id_segment,
                    'concat_curve_segm': concat_curve_segm,
                    'concat_segment_wall': concat_curve_segm_wall,
                    'concat_cardiac_view': concat_curve_cardiac_view
                })
            else:
                concat_id_segment = studyDf['id_position'].values
                concat_traces.append({
                    'study_nr': study_nr,
                    'concatenated_trace': concatenated_trace,
                    'concat_id_segment': concat_id_segment,
                    'concat_curve_segm': concat_curve_segm
                })

        concatenated_tracesDF = pd.DataFrame(concat_traces)
        return concatenated_tracesDF

    def process_study_nr(self, study_nr):
        studyDf = self.segmental_tracesDF[self.segmental_tracesDF['study_nr'] == study_nr]
        tracesDF = pd.DataFrame(studyDf['strain_trace'].apply(safe_eval).tolist())
        concatenated_trace = np.concatenate(tracesDF.values)

        concat_curve_segm = strip_and_get(studyDf['curve_segment']) #.str.rstrip().values

        if not self.isReconstructed:
            concat_curve_len = studyDf['curve_length'].values
            concat_curve_segm_wall = strip_and_get(studyDf['segment_wall']) #.str.rstrip().values
            concat_curve_cardiac_view = strip_and_get(studyDf['cardiac_view']) #.str.rstrip().values
            concat_id_segment = strip_and_get(studyDf['id_position']) #.str.rstrip().values

            return {
                'study_nr': study_nr,
                'concatenated_trace': concatenated_trace,
                'concat_curve_len': concat_curve_len,
                'concat_id_segment': concat_id_segment,
                'concat_curve_segm': concat_curve_segm,
                'concat_segment_wall': concat_curve_segm_wall,
                'concat_cardiac_view': concat_curve_cardiac_view
            }
        else:
            concat_id_segment = studyDf['id_position'].values
            return {
                'study_nr': study_nr,
                'concatenated_trace': concatenated_trace,
                'concat_id_segment': concat_id_segment,
                'concat_curve_segm': concat_curve_segm
            }

    def concatenate_traces_par(self):
        print('Concatenating traces...')
        unique_study_nr = self.segmental_tracesDF['study_nr'].unique()

        # Threads avoid pickling 'self' and are usually faster with pandas/numpy work
        with concurrent.futures.ThreadPoolExecutor() as executor:
            concat_traces = list(executor.map(self.process_study_nr, unique_study_nr, chunksize=10))

        concatenated_tracesDF = pd.DataFrame(concat_traces)
        return concatenated_tracesDF

