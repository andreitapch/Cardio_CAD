import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def runPCA(data):
    twoChTraces = data.twoChTraces
    fourChTraces = data.fourChTraces
    aplaxChTraces = data.aplaxChTraces

    carViewNames = ['2CH', '4CH', 'APLAX']
    carViewCols = [twoChTraces, fourChTraces, aplaxChTraces]

    for carView, cardViewName in zip(carViewCols, carViewNames):
        pca = PCA(n_components=3)
        strainCoords = carView.strainCoords.values.tolist()
        strainCoords = np.array(strainCoords)
        pcaResulSTE = pca.fit_transform(strainCoords)
        # Plot the results
        plotPCA(pcaResulSTE, cardViewName)

        glsCoords = carView.gls.values.tolist()
        glsCoords = np.array(glsCoords)
        pcaResultGls = pca.fit_transform(glsCoords)
        plotPCA(pcaResultGls, cardViewName)


def plotPCA(pcaResult, cardViewName):
    plt.scatter(pcaResult[:, 0], pcaResult[:, 1])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA' + cardViewName)
    plt.show()
