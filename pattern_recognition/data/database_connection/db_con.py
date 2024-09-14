import pandas as pd
import pyodbc

from pattern_recognition.utils.parameters_dics import dbTableNames


class DatabaseConnection:
    def __init__(self, driver, server, database):
        self.driver = driver
        self.server = server
        self.database = database
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = pyodbc.connect(
                f"Driver={self.driver};"
                f"Server={self.server};"
                f"Database={self.database};"
                "Trusted_Connection=yes;"
            )
            self.cursor = self.conn.cursor()
            # print("Connected to the database.")
        except pyodbc.Error as e:
            print("Error connecting to the database:", e)

    def execute_query(self, query):
        try:
            self.cursor.execute(query)
            return self.cursor
        except pyodbc.Error as e:
            print("Error executing the query:", e)
            return None

    def df_query(self):
        try:
            rows = self.cursor.fetchall()
            columns = [column[0] for column in self.cursor.description]
            df = pd.DataFrame.from_records(rows, columns=columns)
            return df
        except pyodbc.Error as e:
            print("Error executing the query:", e)
            return None

    def close_connection(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def insert_values(self, table_name, values):
        try:
            insert_query = f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(values))})"
            self.cursor.execute(insert_query, values)
            self.cursor.commit()
        except pyodbc.Error as e:
            print("Error saving the registry:", e)

    def update_values(self, table_name, column_name, value, condition_column, condition_value):
        try:
            update_query = f"UPDATE {table_name} SET {column_name} = ? WHERE {condition_column} = ?"
            self.cursor.execute(update_query, (value, condition_value))
            self.cursor.commit()
        except pyodbc.Error as e:
            print("Error updating the registry:", e)

    def delete_query(self, table_name):
        delete_query = f"DELETE FROM {table_name}"
        self.cursor.execute(delete_query)
        self.cursor.commit()

    def delete_query_where(self, table_name, column, value):
        delete_query = f"DELETE FROM {table_name} WHERE {column} = {value}"
        self.cursor.execute(delete_query)
        self.cursor.commit()

    def reset_auto_increment(self, table_name):
        try:
            reset_query = f"DBCC CHECKIDENT ('{dbTableNames[table_name]}', RESEED, 0)"
            self.cursor.execute(reset_query)
            self.cursor.commit()
        except pyodbc.Error as e:
            print("Error resetting auto-increment value:", e)
