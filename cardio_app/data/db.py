from contextlib import contextmanager

import pandas as pd
import pyodbc

from cardio_app import settings


def connection_string():
    return (f"Driver={settings.DB['driver']};Server={settings.DB['server']};"
            f"Database={settings.DB['database']};Trusted_Connection=yes;")


@contextmanager
def connect():
    conn = pyodbc.connect(connection_string())
    try:
        yield conn
    finally:
        conn.close()


def read(sql, params=()):
    with connect() as conn:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame.from_records([tuple(row) for row in cursor.fetchall()], columns=columns)


def execute_many(batches):
    with connect() as conn:
        cursor = conn.cursor()
        count = 0
        for sql, rows in batches:
            if rows is None:
                cursor.execute(sql)
            elif rows:
                cursor.executemany(sql, rows)
            count += max(cursor.rowcount, 0)
        conn.commit()
        return count
