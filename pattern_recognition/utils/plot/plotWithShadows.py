import numpy as np
import matplotlib.pyplot as plt


def plotEachSteCurve(segments, stePointsCoords, timePoints, cardViewName, es, color, plotInterpCurves):
    # Convert stePointsCoords to a NumPy array
    stePointsCoords = np.array(stePointsCoords)

    # Calculate the average and standard deviation of stePointsCoords
    average_stePoints = np.mean(stePointsCoords, axis=0)
    std_stePoints = np.std(stePointsCoords, axis=0)

    for stePoints, timePoint, segment in zip(stePointsCoords, timePoints, segments):
        plt.plot(stePoints, color=color['pointColor'], linewidth=1)
        num_ticks = 7
        tick_positions = np.linspace(0, 1, num_ticks)
        tick_labels = [f'{x:.1f}' for x in tick_positions]
        plt.xticks(tick_positions * (len(stePoints) - 1), tick_labels)
        if plotInterpCurves:
            plt.axvline(x=int(es), color=color['lineColor'], linestyle='--')

    # Plot the average curve
    plt.plot(average_stePoints, color='black', linewidth=1, linestyle='--')

    # Plot the shadow representing the variability
    plt.fill_between(range(len(average_stePoints)), average_stePoints - std_stePoints,
                     # average_stePoints + std_stePoints, color='lightblue', alpha=0.3)
                     average_stePoints + std_stePoints, color='mistyrose', alpha=0.3)

    plt.xlabel('ms')
    plt.ylabel('strain %')
    plt.title(cardViewName)
    plt.show()
