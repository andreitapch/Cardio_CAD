import pydicom

def read_dicom_file(filename):
    try:
        dicom_data = pydicom.dcmread(filename)
        return dicom_data
    except Exception as e:
        print(f"An error occurred while reading the DICOM file: {e}")
        return None

dicom_data = read_dicom_file('path_to_your_dicom_file.dcm')