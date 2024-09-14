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
root_dir = 'F:/PHD/DataSets/2D_GE_intervendor_in_vivo'
save_dir = 'D:/PHD/Experiments/Journal_EXP_2022/Datasets/inVivo/train'

project_dir = 'D:/PHD/Development/Python/Cardio_CAD/'
# ------------------------------ Parameters ---------------------------------
mat_file_name = 'USData'
img_mat_name = 'tissue3'
mat_ext = 'mat'
cardiac_view = ['2ch', '4ch']
folder_name = 'Data'
img_name = 'data.'
struct = 'scar_struct'

# --------------------------- Iterate over vendors ------------------------------

for fldr in os.listdir(root_dir):
    for pat_folder in os.listdir(os.path.join(root_dir, fldr, folder_name)):
        for view in cardiac_view:
            file_path = os.path.join(root_dir, fldr, folder_name, pat_folder, view)
            # print(filePath)
            simFrames = read_mat(os.path.join(file_path, (img_name + mat_ext)))
            # USData = simFrames[matFileName]
            # img = USData[imgMatName]
            # imgSize = img.shape

            ## Load scar_struct
            scar_struct_file = os.path.join(root_dir, fldr, folder_name, pat_folder, 'Analysis', (struct + '.' + mat_ext))
            struc = read_mat(scar_struct_file)
            scar_struct = struc[struct]
            # Select sub-structure based on view
            if view == '2ch':
                scar_struct = scar_struct['info_2ch']
            elif view == '4ch':
                scar_struct = scar_struct['info_4ch']

            image_data = simFrames['USData']['tissue3'][:, :,
                        scar_struct['cycle_frame_limits'][0]:scar_struct['cycle_frame_limits'][1]]
            img_size = image_data.shape

            image_data = np.flip(image_data, axis=0)
            image_data = np.transpose(image_data, (1, 0, 2))
            voxelSize = simFrames['USInfo']['voxelsize'][[1, 0, 2]] * 1000

            save_data = os.path.join(save_dir, fldr, pat_folder, view, mat_ext)
            print(save_data)

            os.makedirs(save_data, exist_ok=True)
            for fr in range(0, (img_size[2]+1)):
                img_frame = image_data[:, :, fr]

                frame_3ch = np.stack([img_frame] * 3, axis=-1)
                savemat(os.path.join(save_data, ('frame' + str(fr + 1) + '.' + mat_ext)), {"usFrame": frame_3ch})

print('Done')
