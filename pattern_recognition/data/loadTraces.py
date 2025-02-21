import pandas as pd
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames, dbParameters


class LoadTraces:
    def __init__(self, studyType, table_name, isImputed, isComponents):
        self.isImputed = isImputed
        self.db = None
        self.connect_db()
        if not isComponents:
            self.tracesDF = self.load_and_filter_traces(dbTableNames[table_name], studyType)
        else:
            self.tracesDF, self.study_nr_imputed, self.study_nr_original = (
                self.load_traces_components(dbTableNames[table_name], studyType))

    def load_and_filter_traces(self, table_name, studyType):
        pd.options.display.float_format = '{:.4f}'.format
        tracesDF = self.load_traces(table_name)
        tracesDF = tracesDF.sort_values(by='study_nr')

        tracesDF['studyType'] = tracesDF['studyType'].str.strip()
        filtered_tracesDF = tracesDF[tracesDF[dbParameters['studyType_vName']] == studyType]

        self.isImputed = 1 if self.isImputed is True else (0 if self.isImputed is False else None)

        if table_name in ['speqle_segmentalStrainInterp', 'ste_segmentalStrainInterp', 'phi_segmentalStrainInterp']:
            if self.isImputed == 0:
                study_nr_imputed = filtered_tracesDF.loc[filtered_tracesDF['isImputed'] == 1, 'study_nr'].unique()
                filtered_tracesDF = filtered_tracesDF.loc[~filtered_tracesDF['study_nr'].isin(study_nr_imputed)]
            elif self.isImputed == 1:
                study_nr_imputed = filtered_tracesDF.loc[filtered_tracesDF['isImputed'] == 1, 'study_nr'].unique()
                filtered_tracesDF = filtered_tracesDF.loc[filtered_tracesDF['study_nr'].isin(study_nr_imputed)]
        return filtered_tracesDF

    def load_traces_components(self, table_name, studyType):
        filtered_tracesDF = None
        pd.options.display.float_format = '{:.4f}'.format
        tracesDF = self.load_traces(table_name)
        tracesDF = tracesDF.sort_values(by='study_nr')

        tracesDF['studyType'] = tracesDF['studyType'].str.strip()
        filtered_tracesDF = tracesDF[tracesDF[dbParameters['studyType_vName']] == studyType]

        # List of study numbers with imputed data
        study_nr_imputed = filtered_tracesDF.loc[filtered_tracesDF['isImputed'] == 1, 'study_nr'].unique().tolist()

        # List of original study numbers without imputed data
        filtered_tracesDF_original = filtered_tracesDF.loc[~filtered_tracesDF['study_nr'].isin(study_nr_imputed)]
        study_nr_original = filtered_tracesDF_original['study_nr'].unique().tolist()

        return filtered_tracesDF, study_nr_imputed, study_nr_original

    def connect_db(self):
        dbConParams = dbConParameters
        self.db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        self.db.connect()

    def load_traces(self, table_name):
        self.db.execute_query(sql_queries.select_all_data(table_name))
        tracesDF = self.db.df_query()
        self.db.close_connection()
        return tracesDF
