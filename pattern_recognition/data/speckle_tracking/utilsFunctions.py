import os

def retrieve_folders(directory):
    return [studyNbr_folder for studyNbr_folder in os.listdir(directory) if
            os.path.isdir(os.path.join(directory, studyNbr_folder))]
