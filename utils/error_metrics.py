import numpy as np
from scipy.integrate import trapezoid


def calculate_rrsse(actual, predicted):
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        squared_errors = (actual - predicted) ** 2
        sum_squared_errors = np.sum(squared_errors)
        area_under_curve = trapezoid(np.abs(actual))

        rrsse = np.sqrt(sum_squared_errors) / area_under_curve
        return rrsse
    except Exception as e:
        print("An error occurred: ", e)