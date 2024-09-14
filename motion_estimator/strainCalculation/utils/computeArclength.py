import numpy as np


def arclength(ptsX, ptsY):
    dx = np.diff(ptsX)  # Differences between successive x-coordinates
    dy = np.diff(ptsY)  # Differences between successive y-coordinates
    segment_lengths = np.sqrt(dx ** 2 + dy ** 2)  # Lengths of segments between points
    arclength = np.sum(segment_lengths)  # Sum of segment lengths
    return arclength


def compute_arclength(pts_x, pts_y, isOpenContour, layer):

    if not isOpenContour:
        pts_x = np.append(pts_x, pts_x[:, :1, :], axis=1)
        pts_y = np.append(pts_y, pts_y[:, :1, :], axis=1)

    ptFrames, nFrames = pts_x.shape

    if layer == 'endo':
        layers = [0]
    elif layer == 'mid':
        if nFrames % 2 == 0:  # if even
            layers = [nFrames // 2, nFrames // 2 + 1]
        else:
            layers = [(nFrames + 1) // 2]
    elif layer == 'epi':
        layers = [nFrames - 1]
    elif layer == 'all':
        layers = list(range(nFrames))

    # Compute arclength on each layer
    dl = np.zeros((len(layers), ptFrames))
    for f in range(ptFrames):
        for l in range(len(layers)):
            x = pts_x[:, layers[l]]
            y = pts_y[:, layers[l]]
            dl[l, f] = arclength(x, y)
        stop = 0
    # Compute average arclength (in case there is more than one layer)
    if len(layers) > 1:
        dl = np.mean(dl, axis=0)

    return dl
