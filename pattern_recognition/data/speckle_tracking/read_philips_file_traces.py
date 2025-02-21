import re
import chardet
import pandas as pd
from tqdm import tqdm
import numpy as np
from pyxlsb import open_workbook

from dash_viewer.data_imputation.load_dataTables import load_data_table_db
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, stParameters, dbParameters, dbTableNames, \
    philipsParameters


def del_st_strain_traces(db, tableName):
    db.execute_query(sql_queries.select_delete_all_data(tableName))
    db.cursor.commit()


class PhilipsFileReader:
    def __init__(self, list_st_files, studyType):
        self.studyType = studyType
        print("Reading Strain Traces and storing them in SQL Server...")
        self.db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                      database=dbConParameters['database'])
        self.db.connect()
        del_st_strain_traces(self.db, dbTableNames['phi_segmentalStrain_T'])
        del_st_strain_traces(self.db, dbTableNames['phi_strainValues_T'])
        self.cardiac_segmentsDF = load_data_table_db('config_cardiacSegments_T', False)

        self.read_strain_traces(list_st_files.seg_ste_file_name_all, dbParameters, stParameters, self.studyType)
        self.db.close_connection()

    @staticmethod
    def find_pattern(start_str, end_str, file_path):
        pattern = re.search(re.escape(start_str) + "(.*?)" + re.escape(end_str), file_path)
        return pattern.group(1) if pattern else None

    def read_strain_traces(self, traces_folder_name, dbParameters, st_parameters, study_type):
        strain_data = {}
        AVR_R, MV_R_R, R_AVC = 0, 0, 0
        for traces_filename in tqdm(traces_folder_name):
            file_path = str(traces_filename)

            # if study_type == 'BL':
            study_nbr = self.find_pattern(philipsParameters['philipsAnalysis_baseline'] + '\\', "\\", file_path)
            cardiac_view = self.find_pattern(study_nbr + '\\', philipsParameters['philipsFilePattern'], file_path)
            # print(study_nbr)
            with open(file_path, 'rb') as raw_file:
                raw_data = raw_file.read()
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')

            with open(file_path, 'r', encoding=encoding, errors='replace') as file_content:
                strain_data = {}
                for line in file_content:
                    if 'AV R-R' in line:
                        AVR_R = int(re.search(r'AV\s*R-R\s+(\d+)ms', line).group(1)) / 1000
                    elif 'MV R-R' in line:
                        MV_R_R = int(re.search(r'MV\s*R-R\s+(\d+)ms', line).group(1)) / 1000
                    elif 'R-AVC' in line:
                        R_AVC = int(re.search(r'R-AVC\s+(\d+)ms', line).group(1)) / 1000
                    elif re.search(r'\b\w+\s*Long\. Strain$', line):
                        full_name = line.strip()
                        segment_name = line.strip().split('Long. Strain')[0].strip()
                        strain_data[segment_name] = {}
                        strain_data[segment_name]['txt_name'] = full_name
                        time_vector = []
                        strain_vector = []
                        line = file_content.readline()
                        line = file_content.readline()
                        while line.strip() != '':
                            time, strain = map(float, line.split())
                            time_vector.append(time)
                            strain_vector.append(strain)
                            line = file_content.readline()

                        es_time = time_vector[0] + R_AVC
                        h_rate = 60 / AVR_R
                        n_frames = len(time_vector)

                        strain_data[segment_name]['Time(sec)'] = time_vector
                        strain_data[segment_name]['es_time'] = es_time
                        strain_data[segment_name]['h_rate'] = h_rate
                        strain_data[segment_name]['n_frames'] = n_frames

                        strain_data[segment_name]['L. Strain(%)'] = strain_vector

                first_key = next(iter(strain_data))
                reference_time_vector = strain_data[first_key]['Time(sec)']
                reference_es_time = strain_data[first_key]['es_time']
                reference_h_rate = strain_data[first_key]['h_rate']
                reference_n_frames = strain_data[first_key]['n_frames']

                for key in strain_data:
                    try:
                        assert strain_data[key][
                                   'Time(sec)'] == reference_time_vector, f"Time vectors do not match in file: {file_path} segment: {key}"
                        assert strain_data[key][
                                   'es_time'] == reference_es_time, f"es_time values do not match in file: {file_path} segment: {key}"
                        assert strain_data[key][
                                   'h_rate'] == reference_h_rate, f"h_rate values do not match in file: {file_path} segment: {key}"
                        assert strain_data[key][
                                   'n_frames'] == reference_n_frames, f"n_frames values do not match in file: {file_path} segment: {key}"
                    except AssertionError as e:
                        print(e)
                        raise

                data = {}
                data['ed1_frame'] = np.where(np.array(reference_time_vector) == reference_time_vector[0])[0][0]
                data['es_frame'] = (np.abs(np.array(reference_time_vector) - reference_es_time)).argmin()
                data['ed2_frame'] = np.where(np.array(reference_time_vector) == reference_time_vector[-1])[0][0]
                data['time'] = np.array(reference_time_vector)[data['ed1_frame']:data['ed2_frame'] + 1] - \
                               reference_time_vector[0]

                data['gls'] = {}
                data['gls']['curve'] = np.zeros(data['ed2_frame'] - data['ed1_frame'] + 1)

                for segment_name in strain_data:
                    segment_info = self.cardiac_segmentsDF[self.cardiac_segmentsDF['segment_name_17'] == segment_name]
                    if segment_info.empty:
                        continue
                    if len(segment_info) > 1:
                        pass

                    if cardiac_view.lower() == 'aplax':
                        segment_info = segment_info[
                            (segment_info['cardiac_view'].str.lower() == cardiac_view.lower()) |
                            (segment_info['cardiac_view'].str.lower() == '3ch')]
                    else:
                        segment_info = segment_info[segment_info['cardiac_view'].str.lower() == cardiac_view.lower()]

                    if segment_info.empty:
                        # print('study_nbr:', study_nbr)
                        # print('cardiac_view:', cardiac_view)
                        continue

                    strain_trace = strain_data[segment_name]['L. Strain(%)']

                    segment = segment_info['segment_name'].values[0]
                    id_position = segment_info['id_position'].values[0]
                    segment_wall = segment_info['cardiac_wall'].values[0].lower()
                    data[segment] = {}
                    data[segment]['curve'] = np.array(strain_trace)[data['ed1_frame']:data['ed2_frame'] + 1]
                    data[segment]['curve_txt'] = ', '.join([f"{value:.3f}" for value in data[segment]['curve']])

                    data['gls']['curve'] += data[segment]['curve'] / len(segment_name)
                    data['gls']['curve_txt'] = ', '.join([f"{value:.3f}" for value in data['gls']['curve']])

                    self.db.insert_values(dbTableNames['phi_segmentalStrain_T'],
                                          (study_nbr, segment_wall, int(len(data[segment]['curve'])),
                                           segment, int(id_position),
                                           str(data[segment]['curve_txt']),
                                           cardiac_view, study_type))

                data['ecg'] = None
                data['ed2_frame'] = data['ed2_frame'] - data['ed1_frame'] + 1
                data['es_frame'] = data['es_frame'] - data['ed1_frame']  + 1
                data['ed1_frame'] = 0
                data['ed2_time'] = float(reference_time_vector[-1] - reference_time_vector[0])
                data['es_time'] = reference_es_time - reference_time_vector[0]
                data['ed1_time'] = reference_time_vector[0] - reference_time_vector[0]

                self.insert_values_into_db(dbTableNames, study_nbr, cardiac_view, data, study_type,
                                           segment_wall)

    def insert_values_into_db(self, dbTableNames, study_nbr, cardiac_view, data, study_type,
                              segment_wall):
        self.db.insert_values(dbTableNames['phi_strainValues_T'],
                              (int(study_nbr),
                               cardiac_view, segment_wall, data['ed1_frame'],
                               int(data['es_frame']),
                               int(data['ed2_frame']),
                               str(data['gls']['curve_txt']), str(data['ecg']), data['ed2_time'],
                               data['es_time'], data['ed1_time'], str(data['time']), study_type))
