import os
from tqdm import tqdm
from pattern_recognition.utils.parameters_dics import stParameters, dbParameters
from pattern_recognition.data.speckle_tracking.utilsFunctions import retrieve_folders


def rename_folders(dbParameters, traces_path, traces_folder_name):
    studyNbr_path = retrieve_folders(os.path.join(traces_path, traces_folder_name))
    cardiac_views = dbParameters['cardiac_views']
    for studyNbr in tqdm(studyNbr_path):
        for cardiacView in cardiac_views:
            folderView = str(os.path.join(traces_path, traces_folder_name, studyNbr))
            viewfolderFile = retrieve_folders(folderView)
            for view in viewfolderFile:
                if os.path.exists(folderView):
                    if cardiacView == 'Aplax' and view == 'APLAX':
                        os.rename(os.path.join(folderView, view), os.path.join(folderView, cardiacView))
                    elif cardiacView == '4ch' and view == '4 Ch':
                        os.rename(os.path.join(folderView, view), os.path.join(folderView, cardiacView))
                    elif cardiacView == '2ch' and view == '2ch (1)':
                        os.rename(os.path.join(folderView, view), os.path.join(folderView, cardiacView))


class RenameSTEFolders:
    def __init__(self):
        stPath = stParameters['stPath']

        stAnalysis_all = stParameters['stAnalysis_all']
        stAnalysis_followup = stParameters['stAnalysis_followup']

        rename_folders(dbParameters, stPath, stAnalysis_all)
        rename_folders(dbParameters, stPath, stAnalysis_followup)
