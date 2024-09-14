import numpy as np
import pandas as pd

from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import (dbConParameters, stParameters, dbParameters,
                                                       dbTableNames, SPEQLE_parameters, Cardiac_events)


def extract_parameters():
    parameters = {
        'mean_strain': SPEQLE_parameters['mean_strain_vname'],
        'mean_ecg': SPEQLE_parameters['mean_ecg_vname'],
        'mean_time': SPEQLE_parameters['mean_time_vname'],
        'mean_sr': SPEQLE_parameters['mean_sr_vname'],
        'patient': SPEQLE_parameters['patient_vname'],
        'segment': SPEQLE_parameters['segment_vname'],
        'event_time': SPEQLE_parameters['event_time_vname'],
        'event_length': SPEQLE_parameters['event_length_vname'],
        'curve_segment': SPEQLE_parameters['mean_curves_vname'],
        'id_position': SPEQLE_parameters['id_position_fname'],
        'event_name': SPEQLE_parameters['event_vname'],
        'segment_wall': SPEQLE_parameters['segment_wall_vname'],
        'time_points': SPEQLE_parameters['time_points_vname'],
        'cardiac_view': SPEQLE_parameters['cardiac_view_vname'],
        'curve_length_name': SPEQLE_parameters['curve_length_vname'],
        'ecg': dbParameters['ecg_vName'],
        'gls': dbParameters['gls_vName'],
        'esFrame': dbParameters['esFrame_vName'],
        'ed1Frame': dbParameters['ed1Frame_vName'],
        'ed2Frame': dbParameters['ed2Frame_vName'],
        'ed2Time': dbParameters['ed2Time_vName'],
        'esTime': dbParameters['esTime_vName'],
        'ed1Time': dbParameters['ed1Time_vName'],
        'studyType': dbParameters['studyType_vName']
    }
    return parameters


def load_segment_table_db():
    segment_descT = dbTableNames['config_cardiacSegments_T']
    try:
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.execute_query(sql_queries.select_all_data(segment_descT))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class ReadSpeqleStrainInfo:
    def __init__(self, mean_curves, patient_id, segm_name, event, idx_segment, curve_length, studyType):
        self.studyType = studyType
        self.curves_strain_df = pd.DataFrame()
        self.curves_sr_df = pd.DataFrame()
        self.ecg_df = pd.DataFrame()
        self.events_df = pd.DataFrame()
        self.event_time = []
        self.get_event_timing(event, curve_length, idx_segment)
        self.diastole_length, self.systole_length = [], []
        self.load_curves_df(mean_curves, patient_id, segm_name, event, idx_segment, curve_length)

    def get_event_timing(self, event, curve_length, idx_segment):
        cycle_values = np.linspace(0, 1, num=curve_length)
        # mean_pos = [cycle_values[i - 1] for i in event.event_indices]
        self.event_time = event.event_indices[idx_segment] + [curve_length]
        pass

    def calculate_EdEs(self, curve_length):
        end_systole_fname = SPEQLE_parameters['end_systole']
        end_systole = self.event_time[end_systole_fname]
        self.systole_length = end_systole
        self.diastole_length = curve_length - end_systole

    def load_curves_df(self, mean_curves, patient_id, segm_name, event, idx_segment, curve_length):
        segm_name = segm_name.strip()
        self.calculate_EdEs(curve_length)
        parameters = extract_parameters()

        segment_tableDF = load_segment_table_db()
        filtered_segmentDF = segment_tableDF[segment_tableDF['seg_speqle'].str.lower() == segm_name.lower()]
        segment_name_list = [name for name in filtered_segmentDF['segment_name']]
        cardiac_view_list = [name for name in filtered_segmentDF['cardiac_view']]
        id_position_list = [name for name in filtered_segmentDF['id_position']]
        mean_strain_list = [strain.tolist() for strain in mean_curves[parameters['mean_strain']][0]]

        if len(mean_strain_list) > 1:
            cardiac_view_list = cardiac_view_list
            id_position_list = id_position_list
            segment_name_list = segment_name_list
        else:
            cardiac_view_list = cardiac_view_list[0]
            id_position_list = id_position_list[0]
            segment_name_list = [0]

        mean_sr_list = [sr.tolist() for sr in mean_curves[parameters['mean_sr']][0]]
        cardiac_events_values = [event['Name'] for event in Cardiac_events.values()]

        self.curves_strain_df = pd.DataFrame({
            parameters['patient']: np.repeat(patient_id, len(mean_strain_list)),
            parameters['segment_wall']: np.repeat(segm_name, len(mean_strain_list)),
            parameters['curve_length_name']: np.repeat(curve_length, len(mean_strain_list)),
            parameters['curve_segment']: segment_name_list,
            parameters['id_position']: id_position_list,
            parameters['mean_strain']: mean_strain_list,
            parameters['cardiac_view']: cardiac_view_list,
            parameters['studyType']: np.repeat(self.studyType, len(mean_strain_list))
        })
        self.curves_sr_df = pd.DataFrame({
            parameters['patient']: np.repeat(patient_id, len(mean_sr_list)),
            parameters['segment_wall']: np.repeat(segm_name, len(mean_sr_list)),
            parameters['curve_length_name']: np.repeat(curve_length, len(mean_sr_list)),
            parameters['curve_segment']: segment_name_list,
            parameters['id_position']: id_position_list,
            parameters['mean_sr']: mean_sr_list,
            parameters['cardiac_view']: cardiac_view_list,
            parameters['studyType']: np.repeat(self.studyType, len(mean_sr_list))
        })
        self.events_df = pd.DataFrame({
            parameters['patient']: np.repeat(patient_id, len(self.event_time)),
            parameters['segment']: np.repeat(segm_name, len(self.event_time)),
            parameters['event_name']: cardiac_events_values,
            parameters['event_length']: event.event_length[idx_segment],
            parameters['event_time']: self.event_time,
            parameters['studyType']: np.repeat(self.studyType, len(self.event_time))
        })
        self.ecg_df = pd.DataFrame({
            parameters['patient']: patient_id,
            parameters['cardiac_view']: cardiac_view_list[0],
            parameters['segment']: segm_name,
            parameters['ed1Frame']: 0,
            parameters['esFrame']: self.systole_length,
            parameters['ed2Frame']: self.diastole_length,
            parameters['gls']: None,
            parameters['ecg']: list(mean_curves[parameters['mean_ecg']][0]),
            parameters['ed2Time']: mean_curves[parameters['mean_time']][0][0][self.diastole_length],
            parameters['esTime']: mean_curves[parameters['mean_time']][0][0][self.systole_length],
            parameters['ed1Time']: mean_curves[parameters['mean_time']][0][0][0],
            parameters['time_points']: list(mean_curves[parameters['mean_time']][0]),
            parameters['studyType']: self.studyType
        })
