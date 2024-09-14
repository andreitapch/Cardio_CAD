import pyodbc

from pattern_recognition.utils.utils_db import insert_traces, update_trace_column
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc


class InsertTracesToDB:
    def __init__(self, eventsDf, curvesDf_strain, curvesDf_sr, ecgDf):
        dbConParams = dbConParameters
        self.db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        self.db.connect()

        insert_traces(self.db, ecgDf, 'speqle_strainValues_T', False)
        insert_traces(self.db, curvesDf_strain, 'speqle_segmentalStrain_T', True)
        insert_traces(self.db, curvesDf_sr, 'speqle_segmentalSr_T', True)
        insert_traces(self.db, eventsDf, 'speqle_cardiacEvent_T', True)
        self.db.close_connection()


class InsertInterpTracesToDB:
    def __init__(self, curvesDf_strain, table_name, is_segmental):
        dbConParams = dbConParameters
        self.db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        self.db.connect()
        insert_traces(self.db, curvesDf_strain, table_name, is_segmental)
        self.db.cursor.commit()
        self.db.close_connection()


class DeleteToDB:
    def __init__(self, table_name):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.delete_query(dbTableNames[table_name])
        db.close_connection()


class DeleteToDBWhere:
    def __init__(self, table_name, column, value):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.delete_query_where(dbTableNames[table_name], column, value)
        db.close_connection()


class UpdateTracesToDB:
    def __init__(self, table_name):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.reset_auto_increment(table_name)
        db.close_connection()
