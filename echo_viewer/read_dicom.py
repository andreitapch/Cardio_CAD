import numpy as np
import pandas as pd
import os
import torch
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import pydicom
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt



class DicomReader:
    def __init__(self, filename):
        self.filename = filename
        self.dicom_data = None
        self.img_array = None

    def read_dicom_file(self):
        try:
            self.dicom_data = pydicom.dcmread(self.filename, force=True)

        except Exception as e:
            print(f"An error occurred while reading the DICOM file: {e}")

    def get_dicom_data(self):
        return self.dicom_data

    def get_image_array(self):
        self.img_array = self.dicom_data.pixel_array
        plt.imshow(self.img_array)
        plt.show()


