import numpy as np
import pyodbc

from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames


def createCardiacViewsDF(dataframe):
    dataframe['cardiacView'] = dataframe['cardiacView'].str.strip()
    columns_to_convert = ['strainCoords', 'timePoints', 'gls']
    for column in columns_to_convert:
        dataframe[column] = dataframe[column].str.strip('[]')
        dataframe[column] = dataframe[column].apply(
            lambda x: [float(val) for val in x.split()])
    twoChTraces = dataframe.loc[dataframe['cardiacView'] == '2CH'].copy()
    fourChTraces = dataframe.loc[dataframe['cardiacView'] == '4CH'].copy()
    aplaxChTraces = dataframe.loc[dataframe['cardiacView'] == 'APLAX'].copy()
    return twoChTraces, fourChTraces, aplaxChTraces


def createRemodInterpTraces(dataframe, remod):
    dataframe['cardiacView'] = dataframe['cardiacView'].str.strip()
    columns_to_convert = ['strainCoords', 'timePoints', 'gls']
    for column in columns_to_convert:
        dataframe[column] = dataframe[column].str.strip('[]')
        dataframe[column] = dataframe[column].apply(
            lambda x: [float(val) for val in x.split()])
    twoChTraces = dataframe.loc[dataframe['cardiacView'] == '2CH'].copy()
    fourChTraces = dataframe.loc[dataframe['cardiacView'] == '4CH'].copy()
    aplaxChTraces = dataframe.loc[dataframe['cardiacView'] == 'APLAX'].copy()

    twoChTracesRemod = twoChTraces.loc[twoChTraces['Remoddeling'] == str(remod)]
    fourChTracesRemod = fourChTraces.loc[fourChTraces['Remoddeling'] == str(remod)]
    aplaxChTracesRemod = aplaxChTraces.loc[aplaxChTraces['Remoddeling'] == str(remod)]

    return twoChTracesRemod, fourChTracesRemod, aplaxChTracesRemod


def insert_traces(db, curvesDf_strain, table_name, isSegmental):
    column_names = curvesDf_strain.columns.tolist()
    for idx, row in curvesDf_strain.iterrows():
        if isSegmental:
            values = [str(row[name]) if row[name] is not None and isinstance(row[name], list) else row[name] for name in
                      column_names]
        else:
            values = [
                np.array2string(row[name]) if row[name] is not None and isinstance(row[name], np.ndarray) else row[name]
                for name in column_names]

        db.insert_values(dbTableNames[table_name], values)


def update_trace_column(db, curvesDf_strain, table_name, table_column, isSegmental):
    column_names = curvesDf_strain.columns.tolist()
    for idx, row in curvesDf_strain.iterrows():
        if isSegmental:
            values = [str(row[name]) if row[name] is not None and isinstance(row[name], list) else row[name] for name in
                      column_names]
        else:
            values = [
                np.array2string(row[name]) if row[name] is not None and isinstance(row[name], np.ndarray) else row[name]
                for name in column_names]

        db.update_values(table_name, table_column, values)


