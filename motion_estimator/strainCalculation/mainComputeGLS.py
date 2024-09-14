import os.path

from motion_estimator.strainCalculation.loadPredContours import loadPredictions

# Directories
rootDir = 'F:\PHD\Experiments\Journal_MotionEstimation\Results'
folderName = 'InVivo'  # Martinos , MartinosClutter
resultsPath = os.path.join(rootDir, folderName)

# Variables
predictionType = 'predPoints_unCorr_prop_uni.mat'
isOpenContour = 'True'

if folderName == 'InVivo':
    layer = 'endo'
else:
    layer = 'all'

ptsX, ptsY = loadPredictions(resultsPath, predictionType, isOpenContour, layer)
