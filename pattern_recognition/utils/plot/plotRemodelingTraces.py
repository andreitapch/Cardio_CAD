import numpy as np
from pattern_recognition.utils.utils_db import createRemodInterpTraces
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.plot.plotWithShadows import plotEachSteCurve


# %matplotlib inline

class PlotRemodelingTraces:
    def __init__(self, plotInterpCurves, remod):
        self.remod = remod
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
            db.execute_query(sql_queries.select_remodeling_interp_data())
            SteTracesDF = db.df_query()
        else:
            db.execute_query(sql_queries.select_remodeling_data())
            SteTracesDF = db.df_query()
        self.twoChTraces, self.fourChTraces, self.aplaxChTraces = createRemodInterpTraces(SteTracesDF, self.remod)
        db.close_connection()

    def plotSteCurves(self, plotInterpCurves):
        if plotInterpCurves:
            color = {
                # 'pointColor': 'rosybrown',
                'pointColor': 'lightsteelblue',
                'lineColor': 'navy'}
        else:
            color = {
                'pointColor': 'lightgray',
                'lineColor': 'gray'}

        carViews = ['2CH', '4CH', 'APLAX']
        carViewCols = [self.twoChTraces, self.fourChTraces, self.aplaxChTraces]

        for carView, cardViewName in zip(carViewCols, carViews):
            timePoints = carView.timePoints.values.tolist()
            strainCoords = carView.strainCoords.values.tolist()
            gls = carView.gls.values.tolist()
            segments = carView.segmentName.values.tolist()
            if plotInterpCurves:
                es = int(np.average(carView.esFrame.values.tolist()))
            else:
                es = 0

            plotEachSteCurve(segments, gls, timePoints, 'GLS ' + cardViewName, es, color, plotInterpCurves)
            plotEachSteCurve(segments, strainCoords, timePoints, 'segments ' + cardViewName, es, color,
                             plotInterpCurves)
