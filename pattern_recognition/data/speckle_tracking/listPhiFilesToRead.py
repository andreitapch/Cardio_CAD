import os
import glob
import shutil

import chardet
from tqdm import tqdm


from pattern_recognition.utils.parameters_dics import stParameters, dbParameters, philipsParameters
from pattern_recognition.data.speckle_tracking.utilsFunctions import retrieve_folders


def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    #     print(f'The folder {folder_path} has been deleted.')
    # else:
    #     print(f'The folder {folder_path} does not exist.')

def iterate_available_Phi_info(philipsParameters, traces_path, traces_folder_name, philips_baseline_fldr):
    seg_ste_file_name = []
    missing_ste_files = []
    
    studyNbr_path = retrieve_folders(os.path.join(traces_path, traces_folder_name))
    cardiac_views = philipsParameters['cardiac_views']
    delete_folder(str(os.path.join(traces_path, philips_baseline_fldr)))
    os.makedirs(str(os.path.join(traces_path, philips_baseline_fldr)), exist_ok=True)

    for study_nbr in tqdm(studyNbr_path):
        segTracesPath = str(os.path.join(traces_path, traces_folder_name, study_nbr)) ### TO change xls to txt
        baseline_traces_path = str(os.path.join(traces_path, philips_baseline_fldr, study_nbr))

        if os.path.exists(segTracesPath) and os.listdir(segTracesPath):
            for cardiacView in cardiac_views:
                seg_files_pattern = os.path.join(segTracesPath, cardiacView + '*')
                files = glob.glob(seg_files_pattern)
                for file in files:
                    file_name, file_ext = os.path.splitext(file)
                    if file_ext == '.xls':
                        file_path = os.path.join(segTracesPath, file)
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            encoding = result['encoding']
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                        except UnicodeDecodeError:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()

                        new_file_path = os.path.join(baseline_traces_path, cardiacView + '.txt')
                        os.makedirs(baseline_traces_path, exist_ok=True)
                        with open(new_file_path, 'w', encoding='utf-8') as f:
                            f.write(content)

                        seg_ste_file_name += [new_file_path]
                        # os.remove(file)
                    elif file_ext == '.txt':
                        seg_ste_file_name += [file]
        else:
            os.rmdir(segTracesPath)

    return seg_ste_file_name, missing_ste_files


class ListAvailablePhiFiles:
    def __init__(self):
        self.seg_ste_file_name_all = []
        self.missing_ste_files_all = []
        self.seg_ste_file_name_followup = []
        self.missing_ste_files_followup = []
        phiPath = philipsParameters['philipsPath']
        phiAnalysis_all = philipsParameters['philipsAnalysis_all']
        philips_baseline = philipsParameters['philipsAnalysis_baseline']

        self.seg_ste_file_name_all, self.missing_ste_files_all = (
            iterate_available_Phi_info(philipsParameters, phiPath, phiAnalysis_all, philips_baseline))

        print("Listing Speckle Tracking Traces.... Done ....")
