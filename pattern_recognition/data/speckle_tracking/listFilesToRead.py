import os
import fnmatch
from tqdm import tqdm

from pattern_recognition.utils.parameters_dics import stParameters, dbParameters
from pattern_recognition.data.speckle_tracking.utilsFunctions import retrieve_folders


def iterate_available_ST_info(dbParameters, stParameters, traces_path, traces_folder_name):
    seg_ste_file_name = []
    missing_ste_files = []
    
    studyNbr_path = retrieve_folders(os.path.join(traces_path, traces_folder_name))
    cardiac_views = dbParameters['cardiac_views']

    for study_nbr in tqdm(studyNbr_path):

        for cardiacView in cardiac_views:

            segTracesPath = str(os.path.join(traces_path, traces_folder_name, study_nbr, cardiacView))
            if os.path.exists(segTracesPath):
                seg_files = fnmatch.filter(os.listdir(segTracesPath), f"*{stParameters['stFilePattern']}*")
                if seg_files:
                    seg_ste_file_name += [os.path.join(segTracesPath, file) for file in seg_files]
            else:
                # print(f"{Fore.MAGENTA}Warning: File '{segTracesPath}' not found.{Style.RESET_ALL}")
                missing_ste_files.append(os.path.join(segTracesPath))
                continue

    return seg_ste_file_name, missing_ste_files


class ListAvailableStFiles:
    def __init__(self):
        self.seg_ste_file_name_all = []
        self.missing_ste_files_all = []
        self.seg_ste_file_name_followup = []
        self.missing_ste_files_followup = []
        stPath = stParameters['stPath']
        stAnalysis_all= stParameters['stAnalysis_all']
        stAnalysis_followup = stParameters['stAnalysis_followup']

        self.seg_ste_file_name_all, self.missing_ste_files_all = (
            iterate_available_ST_info(dbParameters, stParameters, stPath, stAnalysis_all))

        self.seg_ste_file_name_followup, self.missing_ste_files_followup = (
            iterate_available_ST_info(dbParameters, stParameters, stPath, stAnalysis_followup))

        print("Listing Speckle Tracking Traces.... Done ....")
