import os
import scipy.io as sio
from motion_estimator.strainCalculation.utils.computeArclength import compute_arclength


def listFolders(path):
    folderName = [f for f in os.listdir(path)
                  if os.path.isdir(os.path.join(path, f))]
    return folderName


def loadPredPoints(seqPath, predictionType, isOpenContour, layer):
    file_path = os.path.join(seqPath, predictionType)
    if not os.path.exists(file_path):
        return None, None
    predPoints = sio.loadmat(file_path)
    predResult = predPoints['predResult']
    ptsY = predResult[0][0][0]
    ptsX = predResult[0][0][1]
    gls = compute_arclength(ptsX, ptsY, isOpenContour, layer)
    return ptsX, ptsY


def loadPredictions(resultsPath, predictionType, isOpenContour, layer):
    tstFolders = listFolders(resultsPath)
    for tFolder in tstFolders:
        foldFolders = listFolders(os.path.join(resultsPath, tFolder))
        for fold in foldFolders:
            modelFolders = listFolders(os.path.join(resultsPath, tFolder, fold))
            for model in modelFolders:
                dataFolders = listFolders(os.path.join(resultsPath, tFolder, fold, model))
                for datafolder in dataFolders:
                    seqFolder = listFolders(os.path.join(resultsPath, tFolder, fold, model, datafolder))
                    for seq in seqFolder:
                        seqPath = os.path.join(resultsPath, tFolder, fold, model, datafolder, seq)
                        # loadPredPoints(seqPath, predictionType, isOpenContour, layer)
                        ptsX, ptsY = loadPredPoints(seqPath, predictionType, isOpenContour, layer)
                        if ptsX is None and ptsY is None:
                            continue
