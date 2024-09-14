# import pandas as pd
#
# from pattern_recognition.data.database_connection.db_con import DatabaseConnection as getDBconnection
# from pattern_recognition.utils.parameters_dics import parameters
#
#
# def read_data_from_excel(file_path):
#     df = pd.read_excel(file_path)
#     return df
#
#
# def read_data_from_database():
#     cursor = getDBconnection()
#     df = pd.DataFrame()
#     return df
#
#
# def select_data_source(file_path):
#     data = None
#     choice = parameters.sourceData
#
#     if choice == 'xlsx':
#         try:
#             data = read_data_from_excel(file_path)
#         except Exception as e:
#             print("Error: ", str(e))
#
#     elif choice == 'db':
#         try:
#             data = read_data_from_database()
#         except Exception as e:
#             print("Error: ", str(e))
#
#     else:
#         print("Invalid choice. Select a Source")
#
#     return data
