import re
from tqdm import tqdm
import numpy as np

from pattern_recognition.data.load_tables import load_data_table_db
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, stParameters, dbParameters, dbTableNames


def del_st_strain_traces(db, tableName):
    db.execute_query(sql_queries.select_delete_all_data(tableName))
    db.cursor.commit()


class StrainTraceFileReader:
    def __init__(self, list_st_files, studyType):
        self.studyType = studyType
        print("Reading Strain Traces and storing them in SQL Server...")
        self.db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                      database=dbConParameters['database'])
        self.db.connect()
        del_st_strain_traces(self.db, dbTableNames['st_segmentalStrain_T'])
        del_st_strain_traces(self.db, dbTableNames['st_strainValues_T'])
        self.cardiac_segmentsDF = load_data_table_db('config_cardiacSegments_T', False)

        self.read_strain_traces(list_st_files.seg_ste_file_name_all, dbParameters, stParameters, self.studyType)
        self.db.close_connection()

    @staticmethod
    def find_pattern(start_str, end_str, file_path, flag):
        sta = str(re.escape(start_str))
        end = str(re.escape(end_str))

        if flag == 0:
            pattern = re.findall(sta + "(.*)" + end, file_path)
            pattern = [match.replace("\\", "") for match in pattern]
            return pattern[0] if pattern else None
        else:
            pattern = re.search(sta + "(.*?)" + end, file_path)
            return pattern.group(1) if pattern else None

    def read_strain_traces(self, traces_folder_name, dbParameters, st_parameters, study_type):
        for traces_filename in tqdm(traces_folder_name):
            file_path = str(traces_filename)
            if study_type == 'BL':
                study_nbr = self.find_pattern(st_parameters['stAnalysis_all'] + '\\', "\\", file_path, 1)
            else:
                study_nbr = self.find_pattern(st_parameters['stAnalysis_followup'] + '\\', "\\", file_path, 1)

            cardiac_view = self.find_pattern(study_nbr, "2DS", file_path, 0)

            with open(file_path, 'r') as file_content:
                file_content.readline()
                num_frames_line = file_content.readline()
                num_frames = int(num_frames_line.replace('Number of Frames ', ''))

                time_line = file_content.readline()
                times = [float(x) if x else 0 for x in re.findall(r'[0-9.]+', time_line)]
                ed1_time, ed2_time, es_time = times[1], times[2], times[3]

                seg_names_line = file_content.readline()
                seg_names = [x.replace(' ', '') for x in re.split(r'\t\t|\t', seg_names_line.strip())][1:-1]

                values = np.full((num_frames, len(seg_names) + 2), np.nan)
                for frame in range(num_frames):
                    line = file_content.readline()
                    values[frame, :] = [float(x.replace(' ', '')) for x in re.split(r'\t\t|\t', line.strip())]
                assert (values.shape[1] - 2) == len(seg_names)
                file_content.close()

                study_nbr = int(study_nbr)

            data = {}

            data['ed1_frame'] = np.where(values[:, 0] == ed1_time)[0][0]
            data['es_frame'] = np.where(values[:, 0] == es_time)[0][0]
            data['ed2_frame'] = np.where(values[:, 0] == ed2_time)[0][0]
            data['time'] = values[data['ed1_frame']:data['ed2_frame'] + 1, 0] - ed1_time

            data['gls'] = {}
            data['gls']['curve'] = np.zeros(data['ed2_frame'] - data['ed1_frame'] + 1)
            for s in range(len(seg_names)):
                segment = seg_names[s]
                segment_info = self.cardiac_segmentsDF[self.cardiac_segmentsDF['segment_name'] == segment]
                id_position = segment_info['id_position'].values[0]
                segment_wall = segment_info['cardiac_wall'].values[0].lower()
                data[segment] = {}
                data[segment]['curve'] = values[data['ed1_frame']:data['ed2_frame'] + 1, s + 1]
                data[segment]['curve_txt'] = ', '.join([f"{value:.6f}" for value in data[segment]['curve']])

                data['gls']['curve'] += data[segment]['curve'] / len(seg_names)
                data['gls']['curve_txt'] = ', '.join([f"{value:.6f}" for value in data['gls']['curve']])


                self.db.insert_values(dbTableNames['st_segmentalStrain_T'],
                                      (study_nbr, segment_wall, int(len(data['gls']['curve'])),
                                       segment, int(id_position),
                                       str(data[segment]['curve_txt']),
                                       cardiac_view, study_type))

            data['ecg'] = values[data['ed1_frame']:data['ed2_frame'] + 1, -1]
            data['ed2_frame'] = data['ed2_frame'] - data['ed1_frame'] + 1
            data['es_frame'] = data['es_frame'] - data['ed1_frame'] + 1
            data['ed1_frame'] = 0
            data['ed2_time'] = float(ed2_time - ed1_time)
            data['es_time'] = es_time - ed1_time
            data['ed1_time'] = ed1_time - ed1_time

            self.insert_values_into_db(dbTableNames, study_nbr, cardiac_view, data, study_type, segment_wall)

        return data

    def insert_values_into_db(self, dbTableNames, study_nbr, cardiac_view, data, study_type, segment_wall):
        self.db.insert_values(dbTableNames['st_strainValues_T'],
                              (study_nbr, cardiac_view, segment_wall, data['ed1_frame'], int(data['es_frame']),
                               int(data['ed2_frame']),
                               str(data['gls']['curve_txt']), str(data['ecg']), data['ed2_time'],
                               data['es_time'], data['ed1_time'], str(data['time']), study_type))
