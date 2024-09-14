import numpy as np


def drift_compensation(x, refframe):
    # Identify and filter valid portion of input
    correct = False
    if np.isnan(x).any():
        valid_frames = np.where(~np.isnan(x))[0]
        x = x[valid_frames]
        refframe = refframe - valid_frames[0] + 1
        correct = True

    # Drift correction
    nframes = x.shape[1]
    y = np.zeros_like(x)
    if refframe == 1:
        y = x - x[:, -1][:, np.newaxis] * ((np.arange(nframes) - 1) / (nframes - 1))
    else:
        y[np.arange(refframe - 1, nframes), :] = x[np.arange(refframe - 1, nframes), :] - x[refframe - 2, :][:,
                                                                                          np.newaxis] * (
                                                             (np.arange(nframes) - 1) / (nframes - 1))
        y[np.arange(refframe - 1), :] = x[np.arange(refframe - 1), :] - x[refframe - 2, :][:, np.newaxis] * (
                    (np.arange(nframes) - 1) / (nframes - 1))

    # Place back in original vector size
    if correct:
        y = np.pad(y, ((0, 0), (valid_frames[0], 0)), mode='constant', constant_values=np.nan)
        y = np.pad(y, ((0, 0), (0, valid_frames[-1])), mode='constant', constant_values=np.nan)

    return y
