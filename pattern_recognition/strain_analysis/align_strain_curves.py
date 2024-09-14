import numpy as np
import pandas as pd

from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.utils.parameters_dics import dbConParameters, dbParameters,dbTableNames
from pattern_recognition.data.database_connection import sql_queries


def find_point_index(stAVCs, stTimePoints):
    stAVCs = stAVCs.values.tolist()
    ptsIndexAvc = []
    for spAVC, stTimePoint in zip(stAVCs, stTimePoints):

        spAVC = round(spAVC, 3)
        stTimePoint = [round(x, 3) for x in stTimePoint]
        try:
            index = stTimePoint.index(spAVC)
            ptsIndexAvc.append(index)
        except ValueError:
            print(spAVC, "not found in the array:")
    return ptsIndexAvc


def interpolate_array(coords, idxToSplit, newIdxFirPart, newIdxSecPart):
    interpArray = None
    traceFirstPart = coords[:idxToSplit]
    traceSecondPart = coords[idxToSplit:]

    x = np.arange(len(traceFirstPart))
    traceInterp1 = np.linspace(0, len(traceFirstPart) - 1, newIdxFirPart)
    trace1Interp = np.interp(traceInterp1, x, traceFirstPart)

    y_origin = np.arange(len(traceSecondPart))
    y_new = newIdxSecPart - len(trace1Interp)
    traceInterp2 = np.linspace(0, len(traceSecondPart) - 1, y_new)
    trace2Interp = np.interp(traceInterp2, y_origin, traceSecondPart)

    interpArray = np.concatenate((trace1Interp, trace2Interp))
    return interpArray


class AlignStrainTraces:
    def __init__(self):
        (self.avgMaxFrames, self.avgAvcFrames,
         self.avgMaxPoints, self.avgAvcTimes) = None, None, None, None
        self.idxAVCList, self.interpTracesDF = None, None
        self.steTracesInterpT = dbTableNames['steTracesInterp_T']
        self.query_call_view()
        self.extract_cardiac_view_df()
        self.calculate_strain_reference_pts()
        self.interpolate_traces()
        self.save_interpolation_values()

    def query_call_view(self):
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        query = sql_queries.select_strain_traces_view()
        db.execute_query(query)
        self.tracesViewDF = db.df_query()
        db.close_connection()

    def extract_cardiac_view_df(self):
        tracesViewDF = self.tracesViewDF.copy()
        tracesViewDF['cardiacView'] = tracesViewDF['cardiacView'].str.strip()
        self.interpTracesDF = pd.DataFrame()
        columns_to_convert = ['strainCoords', 'timePoints', 'gls']
        for column in columns_to_convert:
            tracesViewDF[column] = tracesViewDF[column].str.strip('[]')
            tracesViewDF[column] = tracesViewDF[column].apply(lambda x: [float(val) for val in x.split()])

        self.twoChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == '2ch'].copy()
        self.fourChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == '4ch'].copy()
        self.aplaxChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == 'aplax'].copy()
        pass

    def interpolate_traces(self):
        two_ch_traces = self.twoChTraces
        four_ch_traces = self.fourChTraces
        aplax_ch_traces = self.aplaxChTraces

        carViewCols = [two_ch_traces, four_ch_traces, aplax_ch_traces]  # Info per cardiac view
        carViews = ['2ch', '4ch', 'Aplax']  # Cardiac view

        for cvCol, carView, idxsAVC, avgAvcTime, avgMaxPoint, \
                avgAvcFr, avgMaxFr in zip(carViewCols, carViews, self.idxAVCList, self.avgAvcTimes, self.avgMaxPoints,
                                          self.avgAvcFrames, self.avgMaxFrames):
            traces = cvCol.strainCoords.tolist()
            timePoints = cvCol.timePoints.tolist()
            esFrames = cvCol.esFrame.tolist()
            stGls = cvCol.gls.tolist()
            studyNbrs = cvCol.study_nr.tolist()
            segmentNames = cvCol.segmentName.tolist()

            newEsFrame, newEdFrame, newStudy_Nbr, newCardiacView, newCardiacView = [], [], [], [], []
            newSegmentName, x_interpStrainTraces, y_interpStrainTraces, gls_interpStrainTraces = [], [], [], []

            for stCoord, idxAVC, time, esFrame, gls, \
                    study_Nbr, segmentName in zip(traces, idxsAVC, timePoints, esFrames, stGls, studyNbrs,
                                                  segmentNames):
                newEdFrame.append(avgMaxFr)
                newEsFrame.append(avgAvcFr)
                newStudy_Nbr.append(study_Nbr)
                newSegmentName.append(segmentName)
                newCardiacView.append(carView)

                # X --> Interpolate STE points on X
                x_interpStrainTraces.append(interpolate_array(stCoord, idxAVC, avgAvcTime, avgMaxFr))
                # Y --> Interpolate time points on Y
                y_interpStrainTraces.append(interpolate_array(time, esFrame, avgAvcFr, avgMaxFr))
                # Interpolate GLS
                gls_interpStrainTraces.append(
                    interpolate_array(gls, idxAVC, avgAvcTime, avgMaxFr))  # TODO Fix to ony be do it once per CV

            new_data = {
                'study_Nbr': newStudy_Nbr,
                'cardiacView': newCardiacView,
                'segmentName': newSegmentName,
                'strainCoords': x_interpStrainTraces,
                'timePoints': y_interpStrainTraces,
                'esFrame': newEsFrame,
                'ed2Frame': newEdFrame,
                'gls': gls_interpStrainTraces
            }
            new_data_df = pd.DataFrame(new_data)
            self.interpTracesDF = pd.concat([self.interpTracesDF, new_data_df], ignore_index=True)

        # Reset the index of the DataFrame
        self.interpTracesDF.reset_index(drop=True, inplace=True)

    def calculate_strain_reference_pts(self):
        two_ch_traces = self.twoChTraces
        four_ch_traces = self.fourChTraces
        aplaxChTraces = self.aplaxChTraces

        # To get y points
        # AVC
        avg2chAVC = np.average(two_ch_traces['esFrame'].values.tolist())
        avg4chAVC = np.average(four_ch_traces['esFrame'].values.tolist())
        avgAplaxAVC = np.average(aplaxChTraces['esFrame'].values.tolist())

        # avg ED maximum
        avg2chEd2Fr = np.max(two_ch_traces['ed2Frame'].values.tolist())
        avg4chEd2Fr = np.max(four_ch_traces['ed2Frame'].values.tolist())
        avgAplaxEd2Fr = np.max(aplaxChTraces['ed2Frame'].values.tolist())

        # To get x points
        # Nbr of maximum and average points
        avg2chNpts = np.average(two_ch_traces['pointQty'])
        avg4chNpts = np.average(four_ch_traces['pointQty'])
        avgAplaxNpts = np.average(aplaxChTraces['pointQty'])

        # Nbr of points on AVC Index
        ptsIdx2chAvc = find_point_index(two_ch_traces['esTime'], two_ch_traces['timePoints'])
        ptsIdx4chAvc = find_point_index(four_ch_traces['esTime'], four_ch_traces['timePoints'])
        ptsIdxAplaxAvc = find_point_index(aplaxChTraces['esTime'], aplaxChTraces['timePoints'])

        # Nbr of average points
        avg2chAVCpts = np.average(ptsIdx2chAvc)
        avg4chAVCpts = np.average(ptsIdx4chAvc)
        avgAplaxAVCpts = np.average(ptsIdxAplaxAvc)

        # To interpolate x--> STE points,
        self.idxAVCList = [ptsIdx2chAvc, ptsIdx4chAvc, ptsIdxAplaxAvc]
        self.avgAvcTimes = [int(avg2chAVCpts), int(avg4chAVCpts), int(avgAplaxAVCpts)]
        self.avgMaxPoints = [int(avg2chNpts), int(avg4chNpts), int(avgAplaxNpts)]

        # To interpolate y-->  time points
        self.avgAvcFrames = [int(avg2chAVC), int(avg4chAVC), int(avgAplaxAVC)]
        self.avgMaxFrames = [int(avg2chEd2Fr), int(avg4chEd2Fr), int(avgAplaxEd2Fr)]

    def save_interpolation_values(self):
        study_Nbr = self.interpTracesDF['study_Nbr'].tolist()
        cardiacView = self.interpTracesDF['cardiacView'].tolist()
        segmentName = self.interpTracesDF['segmentName'].tolist()
        strainCoords = self.interpTracesDF['strainCoords']
        timePoints = self.interpTracesDF['timePoints']
        esFrame = self.interpTracesDF['esFrame'].tolist()
        ed2Frame = self.interpTracesDF['ed2Frame'].tolist()
        gls = self.interpTracesDF['gls']

        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.delete_query(self.steTracesInterpT)

        rows = []
        for study, view, segment, strain, time, es, ed, gl in zip(study_Nbr, cardiacView, segmentName, strainCoords,
                                                                  timePoints, esFrame, ed2Frame, gls):
            rows = (study, view, segment, str(strain), str(time), es, ed, str(gl))
            db.insert_values(self.steTracesInterpT, rows)

        db.close_connection()
