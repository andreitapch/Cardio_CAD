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
rootDir = 'D:/PHD/Experiments/Journal_TMI_2022/Datasets/martinoDataset/test_allPy/'
projectDir = 'D:/PHD/Development/Python/Cardio_CAD/'
saveDir = 'D:/PHD/Experiments/Journal_TMI_2022/Datasets/martinoDataset/clutterImgs/'

# ------------------------------ Load Mat ---------------------------------
readTotalInfo = read_mat(projectDir + 'artifact_generation/utils/Totalinfo1.mat')
totalInfo = readTotalInfo['totalinfo']

# ------------------------------ Parameters ---------------------------------
matFileName = 'im_sim'
X_gt = 'X_gt'
matExt = '.mat'
gtName = 'ground_truth'
infoName = 'info'

nArt = 1
aspRto = 8
mwRef = None
movPxlAx = 1

hdList = 210 # 130 #first that worked
vdList = int((hdList - 1) / aspRto) + 1

sigmaH = [45, 23, 25]  # TODO  sd for horizontal # 21 first that worked
sigmaV = [3]# [7 , 13, 3]  # TODO  sd for vertical #3

vel = [0]  # , 0.35, 0.7] #TODO more than 1 vel

gryRange = [180, 200, 255] #brightness


# ------------------------- Create folder function -----------------------------
def create_folders(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# ------------------------- Create Clutter function -----------------------------
def createClutter(img,imgSize, mwRef, maskData, minHeight, maxHeight, halfImg, gtX, gtY, vendorFlr, typeFlr,cardiacView):
    # Start the conversion of the provided MATLAB code to Python
    cnt = 1  # counter of ellipsoid
    for s in range(len(sigmaH)):
        for q in range(len(sigmaV)):
            for secEdg in range(1, 3):  # sector left - right
                # secEdg = 2  # Sector edge: 2 left, 1 right
                for l in range(len(gryRange)):
                    for c in range(0, 3):
                        for v in range(len(vel)):
                            dataArtf = np.zeros((imgSize[0], imgSize[1]), dtype=np.uint8)
                            dataArtfVol = np.zeros_like(img)
                            posArtf = [None] * imgSize[2]
                            # Compute the step of frame and position of artifactual moving for each velocity
                            i = 8
                            j = 5
                            pmwLength = [21, 102]
                            mwLength = pmwLength[1] - pmwLength[0]  # Row number of myocardium wall

                            stp1 = (mwLength * vel[v]) / mwRef
                            if stp1 == 0:
                                stp = 0
                            else:
                                stp = imgSize[2] / stp1  # Step of frame moving

                            for fr in range(0, imgSize[2]):
                                # print(fr)
                                pln = img[:, :, fr]

                                pln_org = pln

                                if fr == 0:  # First frame
                                    sectorGTx = gtX[:, fr]
                                    sectorGTy = gtY[:, fr]

                                    # Compute the rotation angle
                                    p1 = 50
                                    p2 = 100

                                    row1 = np.where(pln_org[p1, :] != 0)[0]  # nonzero
                                    row2 = np.where(pln_org[p2, :] != 0)[0]

                                    pl1 = [p1, min(row1)]
                                    pl2 = [p2, min(row2)]
                                    pr1 = [p1, max(row1)]
                                    pr2 = [p2, max(row2)]

                                    slp_l = (p2 - p1) / (pl1[1] - pl2[1])
                                    slp_r = (p2 - p1) / (pr1[1] - pr2[1])
                                    ang_l = np.rad2deg(np.arctan(slp_l))
                                    ang_r = np.rad2deg(np.arctan(slp_r))
                                    ang_vec = [ang_l, ang_r]

                                    # Select the ellipsoid center randomly
                                    idx_elp = 0
                                    while idx_elp == 0:
                                        # sectorSz = imgSize[0] / 3
                                        sectorSz = (maxHeight - minHeight) / 3
                                        stFirstSec = minHeight
                                        stSecSec = stFirstSec + sectorSz
                                        stThiSec = stSecSec + sectorSz
                                        # sSectorSt = sectorSz
                                        # sSectorEnd = sSectorSt + sectorSz

                                        if secEdg == 1:
                                            mask_x = np.logical_and(sectorGTx >= halfImg, sectorGTx <= imgSize[1])
                                            secEdgstr = '-Right'

                                        else:
                                            mask_x = np.logical_and(sectorGTx >= 0, sectorGTx <= halfImg)
                                            secEdgstr = '-Left'

                                        mask_y = np.logical_and(sectorGTy >= stFirstSec, sectorGTy <= stSecSec)

                                        mask = np.logical_and(mask_y, mask_x)

                                        x = np.array(sectorGTx)[mask]
                                        y = np.array(sectorGTy)[mask]

                                        num_points = 5

                                        # Generate additional points between the original points
                                        filtX = np.interp(np.linspace(0, len(x) - 1, len(x) * num_points),
                                                          np.arange(len(x)), x)
                                        filtY = np.interp(np.linspace(0, len(y) - 1, len(y) * num_points),
                                                          np.arange(len(y)), y)

                                        lenPts = len(filtY)
                                        randomIdx = np.random.choice(lenPts, 1, replace=False)
                                        c1 = round(filtX[randomIdx][0], 4)
                                        r1 = round(filtY[randomIdx][0], 4)

                                        # r1 = 162.3921
                                        # c1 = 413.6998

                                        elpCnt = [r1, c1]

                                        # plt.imshow(pln_org, cmap='gray')
                                        # plt.axhline(y=stFirstSec, color='r')
                                        # plt.axhline(y=stSecSec, color='r')
                                        # plt.axhline(y=stThiSec, color='r')
                                        # plt.axhline(y=stThiSec + sectorSz, color='r')
                                        # plt.axis('off')
                                        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                                        # plt.show()
                                        #
                                        # plt.imshow(pln_org, cmap='gray')
                                        # plt.axhline(y=stFirstSec, color='r')
                                        # plt.axhline(y=stSecSec, color='r')
                                        # plt.scatter(sectorGTx, sectorGTy, color='blue', marker='.')
                                        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                                        # plt.axis('off')
                                        # plt.show()
                                        #
                                        # plt.imshow(pln_org, cmap='gray')
                                        # plt.axhline(y=stFirstSec, color='r')
                                        # plt.axhline(y=stSecSec, color='r')
                                        # plt.scatter(filtX, filtY, color='blue', s=0.5)
                                        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                                        # plt.axis('off')
                                        # plt.show()

                                        if secEdg == 1:  ##Right # Angle of ellipsoid rotation
                                            angRot = [83, 85, 90]

                                        elif secEdg == 2:  ##Left
                                            angRot = [80, 85, 87]

                                        artf_pln, spimp, pos_artf1, pos_artfm = EllipArtGen2(pln_org, elpCnt, hdList,
                                                                                             vdList, secEdg, ang_vec,
                                                                                             sigmaH[s], sigmaV[q],
                                                                                             angRot[c], gryRange[l],
                                                                                             filtX, filtY)

                                        if i == 8 and j == 4:  # These views are exceptions
                                            (sp_pln1, sp_pln2, pos_sp, sp_spimp, edgeout, edgein, edgeoutl, edgeoutr) = \
                                                Edgfind2(pln_org, secEdg, pos_artf1, spimp, artf_pln, imgSize)
                                        elif i == 8 and j == 5:
                                            (sp_pln1, sp_pln2, pos_sp, sp_spimp, edgeout, edgein, edgeoutl, edgeoutr) = \
                                                Edgfind3(pln_org, secEdg, pos_artf1, spimp, artf_pln, imgSize)

                                        prnArtf = spimp

                                        dataArtf = sp_pln2
                                        idx_elp = 1

                                if fr != 0 and stp == 0 and vel[v] == 0:
                                    dataArtf = cv2.add(pln_org, prnArtf)

                                    # dataArtf = cv2.add(pln_org, dataArtf)

                                elif stp != 0 and vel[v] != 0 and fr % stp == 0:
                                    elpCnt[0] += movPxlAx  # moving the ellipsoid center pixel

                                    artf_pln, spimp, posRtf, posArtfm = EllipArtGen2(pln_org, elpCnt,
                                                                                     hdList,
                                                                                     vdList, secEdg,
                                                                                     ang_vec,
                                                                                     sigmaH[s], sigmaV[q],
                                                                                     angRot[c], gryRange[l],
                                                                                     filtX, filtY)

                                    sp_pln2, pos_sp, sp_spimp = Edgfindf(pln_org, secEdg, posRtf, spimp,
                                                                         edgeoutl, edgeoutr)

                                    prnArtf = sp_pln2 - pln_org #TODO
                                    dataArtf = cv2.add(pln_org, prnArtf)

                                dataArtf = dataArtf * maskData
                                dataArtf3d = np.stack([dataArtf] * 3, axis=-1)
                                dataArtfVol[:, :, fr] = dataArtf

                                saveData = os.path.join(saveDir,  vendorFlr, cardiacView,
                                                        typeFlr + '_Vel' + str(vel[v]) + '_gry' + str(
                                                            gryRange[l]) + secEdgstr)
                                frNbr = fr + 1
                                os.makedirs(saveData + '/mat', exist_ok=True)
                                savemat(saveData + '/mat/frame' + str(frNbr) + '.mat', {"usFrame": dataArtf3d})

                                os.makedirs(saveData + '/png', exist_ok=True)
                                image = Image.fromarray(dataArtf)
                                image.save(saveData + '/png/frame' + str(frNbr) + '.png')

                                cnt += 1
                            saveData = os.path.join(saveDir,  vendorFlr,cardiacView,
                                                    typeFlr + '_Vel' + str(vel[v]) + '_gry' + str(
                                                        gryRange[l]) + secEdgstr)
                            saveData = saveData.replace("\\", "/")
                            print(saveData + '/im_sim_clutter' + '.mat')
                            savemat(saveData + '/im_sim_clutter' + '.mat', {"im_sim": dataArtfVol})


def createMask(im_sim):
    maskData = np.sum(im_sim, axis=2)
    maskData = binary_dilation(maskData != 0, structure=np.ones((3, 3)))
    maskData = maskData.astype(bool)


    white_pixel_coords = np.column_stack(np.where(maskData == 1))

    # Sort white pixel coordinates by row
    white_pixel_coords_sorted = white_pixel_coords[np.argsort(white_pixel_coords[:, 0])]

    # Calculate the height of white points
    minHeight = white_pixel_coords_sorted[0][0]
    maxHeight = white_pixel_coords_sorted[-1][0]

    return maskData, minHeight, maxHeight


# --------------------------- Iterate over vendors ------------------------------

folderList = os.listdir(rootDir)
for vendorFlr in folderList: #[4:]: # 4 and A4C misssing
    for cardiacView in os.listdir(os.path.join(rootDir, vendorFlr)):
        # cardiacView = 'A4C'

        for typeFlr in os.listdir(os.path.join(rootDir, vendorFlr, cardiacView)):
            # print(os.path.join(rootDir, vendorFlr, typeFlr, matFileName))

            # load Image, gt and pix spacing
            simFrames = read_mat(os.path.join(rootDir, vendorFlr, cardiacView, typeFlr, (matFileName + matExt)))
            gtSim = read_mat(os.path.join(rootDir, vendorFlr, cardiacView, typeFlr, (gtName + matExt)))
            infoSim = read_mat(os.path.join(rootDir, vendorFlr, cardiacView, typeFlr, (infoName + matExt)))

            info = infoSim['info']  # TODO FIX
            gtSim = gtSim[X_gt]
            img = simFrames[matFileName]

            # plt.imshow(img[:,:,1], cmap='gray')
            # plt.show()

            pixSpacingX = info['x'][1] - info['x'][0]  ## TODO create a function
            pixSpacingY = info['y'][1] - info['y'][0]
            gtX = gtSim[0:36, 0, :] / pixSpacingX + 1
            gtY = gtSim[0:36, 2, :] / pixSpacingX + 1
            # plt.scatter(gtX, gtY, c='blue', marker='o', label='Data')
            # plt.show()
            mwRef = CalculateLVheight(gtX[:, 1], gtY[:, 1])  # LV height in pixels

            # plt.scatter(gtX, gtY, c='blue', marker='o', label='Data')

            imgSize = img.shape
            maskData, minHeight, maxHeight = createMask(img)
            halfImg = imgSize[1] / 2

            # data_org = np.zeros((imgSize[0], imgSize[1], imgSize[2]), dtype=np.uint8)
            createClutter(img,imgSize, mwRef, maskData, minHeight, maxHeight, halfImg, gtX, gtY, vendorFlr, typeFlr, cardiacView)
