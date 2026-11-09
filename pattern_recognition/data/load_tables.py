from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbTableNames, dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc


def load_data_table_db(table_name, isWhere):
    segment_descT = dbTableNames[table_name]
    try:
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        if isWhere:
            db.execute_query(sql_queries.select_all_data(segment_descT),)
        else:
            db.execute_query(sql_queries.select_all_data(segment_descT))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
