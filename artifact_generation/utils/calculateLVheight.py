import math
import numpy as np
from scipy.interpolate import interp1d

def CalculateLVheight(x, y):
    yTop = np.max(y)
    index = np.argmax(y)
    xTop = x[index]

    xMax = np.max(x)
    index = np.argmax(x)
    yMinM = y[index]

    xMin = np.min(x)
    index = np.argmax(x)
    yMin = y[index]

    # Calculate the lengths of the sides
    base = xMax - xMin
    interp_func = interp1d([xMin, xMax], [yMin, yMinM], bounds_error=False, fill_value="extrapolate")
    yBase = interp_func(base / 2)
    perpendicular = yTop - yBase

    hypotenuse = math.sqrt(base ** 2 + perpendicular ** 2) # Calculate the hypotenuse
    # Calculate the height
    height = (base * perpendicular) / hypotenuse

    return height