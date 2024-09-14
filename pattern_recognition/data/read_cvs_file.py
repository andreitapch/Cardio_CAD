import numpy as np


def read_echopac_csv(filename, plot_results=False, drift_compensate=False, segment=False):
    # Parse a 'full store' EchoPac output file

    # Read a 'full store' EchoPac 2D strain CSV export file
    # filename: full path to the file
    # plot_results: boolean (0/1) (default: 0)
    # drift_compensate: boolean (0/1) (default: 0)
    # segment: boolean (0/1) (default: 0) defines whether or not 6 standard segments are returned

    if not segment:
        segment = 0
    if not drift_compensate:
        drift_compensate = 0
    if not plot_results:
        plot_results = 0

    # Open the file
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print('File not found...')
        return None, None, None, None, None, None, None

    # Skip the first couple of lines
    for _ in range(7):
        lines.pop(0)

    # Read some utils on timings
    values = lines.pop(0)
    index1 = values.find('FR=')
    index2 = values.find('Left Marker Time=')
    index3 = values.find('Right Marker Time=')
    index4 = values.find('ES Time=')
    frame_rate = float(values[index1 + 3:index2 - 1])
    tstart = float(values[index2 + 18:index3 - 1])
    tstop = float(values[index3 + 19:index4 - 1])
    tES = float(values[index4 + 9:-1])

    lines.pop(0)  # Skip one line

    # Read the number of knots that have been stored
    dims = [int(num) for num in lines.pop(0).split()]
    knots = np.zeros((dims[0], dims[1], 2))  # Dimensions: time, number, x-y coordinate

    for knot_nr in range(dims[0]):
        values = [float(num) for num in lines.pop(0).split()]
        knots[knot_nr, :, 0] = values[::2]
        knots[knot_nr, :, 1] = values[1::2]

    # find the indices to the tstart, tstop, and tES
    t = np.arange(1, dims[0] + 1) / frame_rate
    tstart_index = np.argmin(np.abs(t - tstart))
    tstop_index = np.argmin(np.abs(t - tstop))
    tES_index = np.argmin(np.abs(t - tES))
    tstrain = t[tstart_index:tstop_index]

    # Calculate the strain
    strain = np.zeros((tstop_index - tstart_index, dims[1]))
    for t_index in range(tstop_index - tstart_index):
        for knot_nr in range(dims[1] - 1):
            e = knots[tstart_index, knot_nr + 1, 0]
