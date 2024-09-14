import os
import glob
import pandas as pd

excel_path = 'F:\\DOPPLER_CIP\\DOPPLER_CIP_clinicalData'
folder_name = ['BASELINE', 'FOLLOWUP']

excel_file_name = 'R_Clinical_Data_Full'
excel_file_name_US = 'v_us_3d_w_outliers'

study_num = ['10008', '10025', '5025', '5028', '5029', '5030', '5031', '5034', '5062', '5067', '5071', '5076',
             '5088', '5091', '5094', '5099', '5124', '5125', '5132', '5149', '5154', '5156', '5157', '5159', '5160',
             '5162', '5173', '5177', '5187', '5204', '5211', '5215', '5230', '5247', '5250', '5254', '5255', '5263',
             '5274', '5285', '5316', '5332', '5337', '5348', '5351', '5371', '5403', '5409', '5424', '5427', '5434',
             '5443', '5485', '5520', '5552', '5554', '5576', '5651', '5667', '5670', '5672', '5678', '5680', '5683',
             '5684', '5687', '5689', '5697', '5698', '5703', '5708', '5723', '5724', '5726', '5738', '5743', '5745',
             '5756', '5758', '5765', '5769', '5782', '5789', '5796', '5801', '5803', '5816', '5819', '5842', '5850',
             '5868', '5901', '5904', '5910', '5918', '5927', '5933', '5937', '5939', '5940', '5990', '5992', '5996',
             '6011', '6013', '6014', '6015', '6019', '6034', '6041', '6048', '6154', '6171', '6175', '6181', '6184',
             '6200', '6229', '6236', '6266', '6358', '6443', '6445', '6481', '6484', '6488', '6506', '6523', '6527',
             '6540', '6549', '6564', '6594', '6610', '6612', '6616', '6627', '6785', '6801', '6842', '6844', '6851',
             '6852', '6856', '6861', '6863', '6866', '6868', '6872', '6875', '6885', '6886', '6889', '6892', '6895',
             '6899', '6907', '6914', '6921', '6935', '6937', '6940', '6941', '6990', '6993', '6995', '6997']

ids_to_find = [int(num) for num in study_num]


def load_save_data(excel_path, folder_name, excel_file, col_to_find,isUS_data,ids_to_find,patient_ids):
    filtered_dfs = pd.DataFrame()
    saved_fname = '_filtered.xlsx'
    for fname in folder_name:
        files = glob.glob(os.path.join(excel_path, fname, excel_file + '*.xlsx'))
        for file in files:
            df = pd.read_excel(file)
            filtered_df = df[df[col_to_find].isin(ids_to_find)]
            filtered_dfs = pd.concat([filtered_dfs, filtered_df])
            if isUS_data:
                patient_ids.extend(filtered_df['PatientID'].tolist())

    filtered_dfs.to_excel(
        os.path.join(excel_path, excel_file + '-' + saved_fname),
        index=False)
    return patient_ids

patient_ids = []
patient_ids = load_save_data(excel_path, folder_name, excel_file_name_US, 'StudyID',True,ids_to_find,patient_ids)
print(len(patient_ids))
patient_ids = load_save_data(excel_path, folder_name, excel_file_name, 'Patient_ID',False,patient_ids,patient_ids)
