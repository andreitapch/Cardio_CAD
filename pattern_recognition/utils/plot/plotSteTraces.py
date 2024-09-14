import numpy as np
import matplotlib.pyplot as plt
from pattern_recognition.utils.utils_db import createCardiacViewsDF
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc


# %matplotlib inline

def plotEachSteCurve(segments, stePointsCoords, timePoints, cardViewName, es, color, plotInterpCurves):
    for stePoints, timePoint, segment in zip(stePointsCoords, timePoints, segments):

        if plotInterpCurves:
            plt.plot(stePoints, color=color['pointColor'], linewidth=1)
            num_ticks = 7
            tick_positions = np.linspace(0, 1, num_ticks)
            tick_labels = [f'{x:.1f}' for x in tick_positions]
            plt.xticks(tick_positions * (len(stePoints) - 1), tick_labels)
            plt.axvline(x=int(es), color=color['lineColor'], linestyle='--')

        else:
            plt.plot(timePoint, stePoints, color=color['pointColor'], linewidth=1)

    plt.xlabel('ms')
    plt.ylabel('strain %')
    plt.title(cardViewName)
    plt.show()



class PlotSteTraces:
    def __init__(self, plotInterpCurves):
        self.aplaxChTracesInterp = None
        self.fourChTracesInterp = None
        self.twoChTracesInterp = None
        self.aplaxChTraces = None
        self.fourChTraces = None
        self.twoChTraces = None
        self.strainTracesViewDF = None
        self.SteTracesInterpTableDF = None
        self.readQueriesToPlot(plotInterpCurves)
        self.plotSteCurves(plotInterpCurves)

    def readQueriesToPlot(self, plotInterpCurves):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        if plotInterpCurves:
            db.execute_query(sql_queries.select_ste_traces_interp_table())
            SteTracesDF = db.df_query()
        else:
            db.execute_query(sql_queries.select_strain_traces_view())
            SteTracesDF = db.df_query()
        self.twoChTraces, self.fourChTraces, self.aplaxChTraces = createCardiacViewsDF(SteTracesDF)
        db.close_connection()

    def plotSteCurves(self, plotInterpCurves):
        if plotInterpCurves:
            color = {
                'pointColor': 'lightGreen',
                'lineColor': 'Green'}
        else:
            color = {
                'pointColor': 'lightBlue',
                'lineColor': 'Blue'}

        twoChTraces = self.twoChTraces
        fourChTraces = self.fourChTraces
        aplaxChTraces = self.aplaxChTraces

        carViews = ['2CH', '4CH', 'APLAX']
        carViewCols = [twoChTraces, fourChTraces, aplaxChTraces]

        for carView, cardViewName in zip(carViewCols, carViews):
            timePoints = carView.timePoints.values.tolist()
            strainCoords = carView.strainCoords.values.tolist()
            gls = carView.gls.values.tolist()
            segments = carView.segmentName.values.tolist() # segmentName segments
            if plotInterpCurves:
                es = int(np.average(carView.esFrame.values.tolist()))
            else:
                es = 0
            plotEachSteCurve(segments, gls, timePoints, 'GLS ' + cardViewName, es, color, plotInterpCurves)
            plotEachSteCurve(segments, strainCoords, timePoints, 'segments ' + cardViewName, es, color,
                             plotInterpCurves)
