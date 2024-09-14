import os
import h5py

from echo_viewer.read_dicom import DicomReader
from echo_viewer.read_h5 import H5Loader

isDICOM = False
isH5 = True

images_path = 'D:\\PHD\\DataSets\\Imram Flash Drive\\CHH-04-001\\'
image_name = 'CHH-04-001-Rest-2C-TDI.h5'

img_file = os.path.join(images_path, image_name)
if isDICOM:
    reader = DicomReader('path_to_your_dicom_file.dcm')
    reader.read_dicom_file()
    dicom_data = reader.get_dicom_data()
    img_array = reader.get_image_array()
else:

    image_name = 'CHH-04-001-5mcg-2C-TDI.h5'
    file_path = os.path.join(images_path, image_name)
    loader = H5Loader(file_path)
    signal_indices = loader.find_dataset_indices()
    # load_tissue = loader.load_file()
    echo_img = loader.load_tissue()
pass