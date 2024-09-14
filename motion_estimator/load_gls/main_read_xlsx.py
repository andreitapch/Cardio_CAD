import os.path

from motion_estimator.strainCalculation.loadPredContours import loadPredictions

# Directories
rootDir = 'F:\PHD\Experiments\Journal_2023\Results'
folderName = 'Intervendor'  # Martinos , MartinosClutter
resultsPath = os.path.join(rootDir, folderName)

# Variables
isOpenContour = 'True'

if folderName == 'Intervendor':
    layer = 'endo'
else:
    layer = 'all'

ptsX, ptsY = loadPredictions(resultsPath, predictionType, isOpenContour, layer)
