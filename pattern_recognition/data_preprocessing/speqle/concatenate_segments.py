import numpy as np


class SegmentsConcatenator:
    def __init__(self, segmental_tracesDF):
        self.segmental_tracesDF = segmental_tracesDF

    def concatenate_segment_traces(self):
        print('grouping traces by segment...')
        self.segmental_tracesDF['curve_segment'] = self.segmental_tracesDF['curve_segment'].str.strip()
        dropna_tracesDF = self.segmental_tracesDF.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[
            ~(dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0) if x is not None else True))]
        grouped = dropna_tracesDF.groupby('study_nr')
        valid_study_df = grouped.filter(lambda x: len(x) == 18)
        valid_study_df = valid_study_df.reset_index(drop=True)
        return valid_study_df

