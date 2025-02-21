import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from dash_viewer.data_imputation.load_dataTables import load_data_table_db
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, dbParameters, dbTableNames


def find_point_index(stAVCs, stTimePoints):
    stAVCs = stAVCs.values.tolist()
    ptsIndexAvc = []
    for spAVC, stTimePoint in zip(stAVCs, stTimePoints):
        spAVC = round(spAVC, 3)
        stTimePoint = [round(x, 3) for x in stTimePoint]
        try:
            index = stTimePoint.index(spAVC)
            ptsIndexAvc.append(index)
        except ValueError:
            print(spAVC, "not found in the array:")
    return ptsIndexAvc


def interpolate_array(coords, idxToSplit, newIdxFirPart, target_total_points):
    traceFirstPart = coords[:idxToSplit]
    traceSecondPart = coords[idxToSplit:]
    len_first_part = len(traceFirstPart)

    remaining_points = target_total_points - newIdxFirPart
    remaining_points = max(remaining_points, 1)

    x_first = np.arange(len_first_part)
    traceInterp1 = np.linspace(0, len_first_part - 1, newIdxFirPart)
    trace1Interp = np.interp(traceInterp1, x_first, traceFirstPart)

    x_second = np.arange(len(traceSecondPart))
    traceInterp2 = np.linspace(0, len(traceSecondPart) - 1, remaining_points)
    trace2Interp = np.interp(traceInterp2, x_second, traceSecondPart)

    interpArray = np.concatenate((trace1Interp, trace2Interp))

    # Optional: Set the first and last points to 0 (if desired)
    interpArray[0] = 0
    interpArray[-1] = 0
    return interpArray


def interpolate_array_(coords, idxToSplit, newIdxFirPart, newIdxSecPart):
    traceFirstPart = coords[:idxToSplit]
    traceSecondPart = coords[idxToSplit:]

    x = np.arange(len(traceFirstPart))
    traceInterp1 = np.linspace(0, len(traceFirstPart) - 1, newIdxFirPart)
    trace1Interp = np.interp(traceInterp1, x, traceFirstPart)

    y_origin = np.arange(len(traceSecondPart))
    y_new = newIdxSecPart - len(trace1Interp)
    traceInterp2 = np.linspace(0, len(traceSecondPart) - 1, y_new)
    trace2Interp = np.interp(traceInterp2, y_origin, traceSecondPart)

    interpArray = np.concatenate((trace1Interp, trace2Interp))
    interpArray[0] = 0
    interpArray[-1] = 0

    return interpArray


def fix_spaces_df(df, param_1, param_2, param_3, cv):
    tracesViewDF = df.copy()
    if cv is not None:
        tracesViewDF[cv] = tracesViewDF[cv].str.strip()
    columns_to_convert = [param_1, param_2, param_3]
    for column in columns_to_convert:
        if column is not None:
            if tracesViewDF[column] is not None:
                tracesViewDF[column] = tracesViewDF[column].str.strip('[]')
                tracesViewDF[column] = tracesViewDF[column].apply(
                    lambda x: [float(val.rstrip(',')) for val in x.split()] if x != 'None' else None)
    strainTracesDF = tracesViewDF.copy()

    return strainTracesDF


class AlignStrainTraces:
    def __init__(self, selected_source_data, interpSelection):
        self.steTracesInterpDF = None
        self.steSegmentalTracesInterpDT = None
        self.ptInterp_esFrame = None
        self.ptInterp_timePoints = None
        self.max_es_frame = None
        self.interpSelection = interpSelection
        self.selected_source_data = selected_source_data
        self.strainTracesDF = pd.DataFrame()

        if self.selected_source_data == 'ST':
            self.strain_table_name = 'st_strainValuesInterp_T'
            self.segmental_table_name = 'st_segmentalStrainInterp_T'
            steTracesT = load_data_table_db('st_strainValues_T', False)
            steTracesSegT = load_data_table_db('st_segmentalStrain_T', False)

        elif self.selected_source_data == 'Philips':
            self.strain_table_name = 'phi_strainValuesInterp_T'
            self.segmental_table_name = 'phi_segmentalStrainInterp_T'
            steTracesT = load_data_table_db('phi_strainValues_T', False)
            steTracesSegT = load_data_table_db('phi_segmentalStrain_T', False)
        else:
            print('No source data selected')
            return

        self.steTracesDF = fix_spaces_df(steTracesT, 'gls', 'timePoints', 'ecg', 'cardiacView')
        self.steSegmentalTracesDT = fix_spaces_df(steTracesSegT, 'strain_trace', None, None, 'cardiac_view')

        self.calculate_strain_reference_pts()
        self.interpolate_traces()
        self.save_interpolation_values()

    def interpolate_traces(self):
        self.steSegmentalTracesInterpDT = self.steSegmentalTracesDT.copy()
        self.steSegmentalTracesInterpDT[['strain_trace', 'curve_length']] = np.nan
        self.steTracesInterpDF = self.steTracesDF.copy()
        self.steTracesInterpDF[['esFrame', 'timePoints', 'ecg', 'gls']] = np.nan
        studyNr_toDelete = []
        print(f'Interpolating {self.interpSelection} traces......')
        for index, st_listDf in self.steTracesDF.iterrows():
            end_systole_frame = st_listDf['esFrame']
            pts_len = len(st_listDf['timePoints'])

            if end_systole_frame == pts_len:
                studyNr_toDelete.append(st_listDf['study_nr'])
                continue

            filtered_df = self.steSegmentalTracesDT[
                (self.steSegmentalTracesDT['study_nr'] == st_listDf['study_nr']) &
                (self.steSegmentalTracesDT['cardiac_view'] == st_listDf['cardiacView'])]

            for idx, st_segListDf in filtered_df.iterrows():
                strain_traces = [float(i) for i in st_segListDf['strain_trace']]

                interpolated_strain = interpolate_array(strain_traces, end_systole_frame, self.ptInterp_esFrame,
                                                        self.ptInterp_timePoints)
                self.steSegmentalTracesInterpDT.at[idx, 'curve_length'] = len(interpolated_strain)

                interpolated_strain = ', '.join([f"{value:.6f}" for value in interpolated_strain])

                self.steSegmentalTracesInterpDT['strain_trace'] = self.steSegmentalTracesInterpDT[
                    'strain_trace'].astype(object)
                # print(idx)
                self.steSegmentalTracesInterpDT.at[idx, 'strain_trace'] = interpolated_strain

            self.steTracesInterpDF.at[index, 'esFrame'] = self.ptInterp_esFrame

            time_points = [float(i) for i in st_listDf['timePoints']]
            interpolated_time_points = interpolate_array(time_points, end_systole_frame, self.ptInterp_esFrame,
                                                         self.ptInterp_timePoints)
            interpolated_time_points = ', '.join([f"{value:.6f}" for value in interpolated_time_points])

            if st_listDf['ecg'] is not None:
                ecg = [float(i) for i in st_listDf['ecg']]
                interpolated_ecg = interpolate_array(ecg, end_systole_frame, self.ptInterp_esFrame,
                                                     self.ptInterp_timePoints)
                interpolated_ecg = ', '.join([f"{value:.6f}" for value in interpolated_ecg])
            else:
                interpolated_ecg = None

            gls = [float(i) for i in st_listDf['gls']]
            interpolated_gls = interpolate_array(gls, end_systole_frame, self.ptInterp_esFrame,
                                                 self.ptInterp_timePoints)
            interpolated_gls = ', '.join([f"{value:.6f}" for value in interpolated_gls])

            self.steTracesInterpDF['timePoints'] = self.steTracesInterpDF[
                'timePoints'].astype(object)
            self.steTracesInterpDF.at[index, 'timePoints'] = str(interpolated_time_points)

            self.steTracesInterpDF['ecg'] = self.steTracesInterpDF['ecg'].astype(object)
            self.steTracesInterpDF.at[index, 'ecg'] = str(interpolated_ecg)

            self.steTracesInterpDF['gls'] = self.steTracesInterpDF['gls'].astype(object)
            self.steTracesInterpDF.at[index, 'gls'] = str(interpolated_gls)

        self.steTracesInterpDF['isImputed'] = 0
        self.steSegmentalTracesInterpDT['isImputed'] = 0

        self.steTracesInterpDF = self.steTracesInterpDF[~self.steTracesInterpDF['study_nr'].isin(studyNr_toDelete)]
        self.steSegmentalTracesInterpDT = self.steSegmentalTracesInterpDT[
            ~self.steSegmentalTracesInterpDT['study_nr'].isin(studyNr_toDelete)]

    def calculate_strain_reference_pts(self):
        if self.selected_source_data == 'ST':
            if self.interpSelection == 'max':
                self.ptInterp_esFrame = int(np.max(self.steTracesDF['esFrame'].values.tolist()))
                self.ptInterp_timePoints = int(np.max(self.steSegmentalTracesDT['curve_length'].values.tolist()))
            elif self.interpSelection == 'avg':
                self.ptInterp_esFrame = int(np.mean(self.steTracesDF['esFrame'].values.tolist()))
                self.ptInterp_timePoints = int(np.mean(self.steSegmentalTracesDT['curve_length'].values.tolist()))

        elif self.selected_source_data == 'Philips':
            table_name = 'st_segmentalStrainInterp_T'
            ste_segmental_interp_t = load_data_table_db(table_name, False)
            table_name = 'st_strainValuesInterp_T'
            steTracesT = load_data_table_db(table_name, False)

            if self.interpSelection == 'max':
                self.ptInterp_esFrame = int(np.max(steTracesT['esFrame'].values.tolist()))
                self.ptInterp_timePoints = int(np.max(ste_segmental_interp_t['curve_length'].values.tolist()))
            elif self.interpSelection == 'avg':
                self.ptInterp_esFrame = int(np.mean(steTracesT['esFrame'].values.tolist()))
                self.ptInterp_timePoints = int(np.mean(ste_segmental_interp_t['curve_length'].values.tolist()))

    def save_interpolation_values(self):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.delete_query(dbTableNames[self.strain_table_name])
        db.delete_query(dbTableNames[self.segmental_table_name])

        print('Saving strain traces......')
        try:
            for index, row in tqdm(self.steTracesInterpDF.iterrows(), total=self.steTracesInterpDF.shape[0]):
                values = row.tolist()
                db.insert_values(dbTableNames[self.strain_table_name], values)

            print('Saving segmental strain traces......')
            for index, row in tqdm(self.steSegmentalTracesInterpDT.iterrows(),
                                   total=self.steSegmentalTracesInterpDT.shape[0]):
                values = row.tolist()
                db.insert_values(dbTableNames[self.segmental_table_name], values)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            db.close_connection()
