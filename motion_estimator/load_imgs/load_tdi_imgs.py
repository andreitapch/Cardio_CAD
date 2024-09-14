import os
import h5py
import glob
# from dicom2hdf import *


# Path
file_path = "D:/PHD/DataSets/Imram Flash Drive/"
tissue_doppler_data = True
file_pattern = "TDI"
def load_h5_filenames(file_path, pattern):
    if tissue_doppler_data:
        h5_files = glob.glob(f"{file_path}*{pattern}.h5")
    else:
        h5_files = glob.glob(f"{file_path}*.h5")
    return h5_files


def load_img_data(h5_file_paths, us_data=None):
    img_data = []
    for h5_file_path in h5_file_paths:
        try:
            with h5py.File(h5_file_path, 'r') as file:
                key_list = list(file.keys())
                for key in key_list:
                    data_key = file[key][:]
                    us_data[key] = data_key
                img_data.append(us_data)
        except Exception as e:
            print(f"Error encountered while extracting tissue Doppler data from {h5_file_path}: {e}")
    return img_data

file_pathS = load_h5_filenames(file_path, file_pattern)
tissue_doppler_data = load_img_data(file_pathS)

if tissue_doppler_data is not None:
    print(tissue_doppler_data)
else:
    print("Tissue Doppler data extraction failed.")
