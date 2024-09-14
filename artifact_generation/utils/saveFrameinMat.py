import os
import cv2
import numpy as np
from PIL import Image
from scipy.io import savemat
import matplotlib.image as img
import matplotlib.pyplot as plt
from pymatreader import read_mat
from scipy.ndimage import binary_dilation

from artifact_generation.edgfind2 import Edgfind2
from artifact_generation.edgfind3 import Edgfind3
from artifact_generation.edgfindf import Edgfindf
from artifact_generation.ellipArtGen2 import EllipArtGen2
from artifact_generation.utils.calculateLVheight import CalculateLVheight

# ------------------------------ Path to files ------------------------------
rootDir = 'D:/PHD/Experiments/Journal_TMI_2022/Datasets/martinoDataset/test/'
projectDir = 'D:/PHD/Development/Python/Cardio_CAD/'
# ------------------------------ Parameters ---------------------------------
matFileName = 'im_sim'
matExt = '.mat'



# --------------------------- Iterate over vendors ------------------------------
for vendorFlr in os.listdir(rootDir):
    for cardiacView in os.listdir(os.path.join(rootDir, vendorFlr)):
        for patType in os.listdir(os.path.join(rootDir, vendorFlr, cardiacView)):
            print(os.path.join(rootDir, vendorFlr, cardiacView, patType, (matFileName + matExt)))
            simFrames = read_mat(os.path.join(rootDir, vendorFlr, cardiacView, patType, (matFileName + matExt)))
            img = simFrames[matFileName]
            imgSize = img.shape
            saveData = os.path.join(rootDir, vendorFlr, cardiacView, patType, 'mat')
            os.makedirs(saveData, exist_ok=True)
            for fr in range(0, imgSize[2]):
                imgFrame = img[:, :, fr]

                frame3ch = np.stack([imgFrame] * 3, axis=-1)

                savemat(os.path.join(saveData, ('frame' + str(fr+1) + matExt)), {"usFrame": frame3ch})
        stop = 1
print('Done')