import pandas as pd
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames, dbParameters


class LoadTracesSpeqle:
    def __init__(self, studyType, table_name, isImputed):
        self.isImputed = isImputed
        self.db = None
        self.connect_db()
        self.tracesDF = self.load_and_filter_traces(dbTableNames[table_name], studyType)

    def load_and_filter_traces(self, table_name, studyType):
        tracesDF = self.load_traces(table_name)
        tracesDF['studyType'] = tracesDF['studyType'].str.strip()
        filtered_tracesDF = tracesDF[tracesDF[dbParameters['studyType_vName']] == studyType]
        if table_name == 'speqle_segmentalStrainInterp' and self.isImputed == 0:
            filtered_tracesDF = filtered_tracesDF.loc[filtered_tracesDF['isImputed'] == 0]
        return filtered_tracesDF

    def connect_db(self):
        dbConParams = dbConParameters
        self.db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        self.db.connect()

    def load_traces(self, table_name):
        self.db.execute_query(sql_queries.select_all_data(table_name))
        tracesDF = self.db.df_query()
        self.db.close_connection()
        return tracesDF
