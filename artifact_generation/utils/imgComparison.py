import scipy.io
import numpy as np

# Load the .mat files
origImg = scipy.io.loadmat('D:\\PHD\\DataSets\\database_2d_simulations_uffc\\database_2d_simulations_uffc\\data\\ESAOTE\\A2C\\laddist\\im_sim.mat')['im_sim']
pythonImg = scipy.io.loadmat('D:\\PHD\\Experiments\\Journal_TMI_2022\\Datasets\\martinoDataset\\test\\ESAOTE\\A2C\\laddist\\im_sim.mat')['im_sim']

# Perform subtraction operation
substract = origImg - pythonImg

# Calculate and display the maximum, minimum, and mean of the subtracted data
max_val = np.max(substract)
min_val = np.min(substract)
meanOrig = np.mean(origImg)
meanPython = np.mean(pythonImg)

print('Image substraction Max: ', max_val)
print('Image substraction Min: ', min_val)
print('Mean original Img: ', meanOrig)
print('Mean Python Img: ', meanPython)
