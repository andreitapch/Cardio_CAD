import plotly.express as px
import plotly.graph_objects as go
# from pattern_recognition.utils.utils import load_ecg
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, stParameters, dbParameters, dbTableNames


def load_ecg(dataframe):
    dataframe['cardiacView'] = dataframe['cardiacView'].str.strip()
    columns_to_convert = ['timePoints', 'ecg']
    for column in columns_to_convert:
        dataframe[column] = dataframe[column].str.strip('[]')
        dataframe[column] = dataframe[column].apply(
            lambda x: [float(val) for val in x.split()])
    return dataframe


def select_all_from_table(table_name):
    try:
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.execute_query(sql_queries.select_all_data(table_name))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class PlotEcg:
    def __init__(self):
        strainValuesT = dbTableNames['strainValues_T']
        studyNr = dbParameters['studyNr']
        SteTracesDF = select_all_from_table(strainValuesT)
        traces_df = load_ecg(SteTracesDF)
        self.plot_ecg_curves(traces_df, studyNr)


    def plot_ecg_curves(self, traces_df, studyNr):
        for index, row in traces_df.iterrows():
            ecg = row['ecg']
            time_points = row['timePoints']
            study_nr = row['study_nr']
            cardiac_view = row['cardiacView']
            fig = go.Figure(data=go.Scatter(x=time_points, y=ecg))
            fig.update_layout(title=f'ECG Curve for study_nbr {study_nr} cardiac: view{cardiac_view}', xaxis_title='Time Points',
                              yaxis_title='ECG')
            fig.show()
            pass

