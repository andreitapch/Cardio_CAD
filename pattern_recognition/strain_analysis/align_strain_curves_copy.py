# import numpy as np
# import pandas as pd
#
# from dash_viewer.data_imputation.load_dataTables import load_data_table_db
# from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
# from pattern_recognition.utils.parameters_dics import dbConParameters, dbParameters, dbTableNames
# from pattern_recognition.data.database_connection import sql_queries
#
#
# def find_point_index(stAVCs, stTimePoints):
#     stAVCs = stAVCs.values.tolist()
#     ptsIndexAvc = []
#     for spAVC, stTimePoint in zip(stAVCs, stTimePoints):
#
#         spAVC = round(spAVC, 3)
#         stTimePoint = [round(x, 3) for x in stTimePoint]
#         try:
#             index = stTimePoint.index(spAVC)
#             ptsIndexAvc.append(index)
#         except ValueError:
#             print(spAVC, "not found in the array:")
#     return ptsIndexAvc
#
#
# def interpolate_array(coords, idxToSplit, newIdxFirPart, newIdxSecPart):
#     interpArray = None
#     traceFirstPart = coords[:idxToSplit]
#     traceSecondPart = coords[idxToSplit:]
#
#     x = np.arange(len(traceFirstPart))
#     traceInterp1 = np.linspace(0, len(traceFirstPart) - 1, newIdxFirPart)
#     trace1Interp = np.interp(traceInterp1, x, traceFirstPart)
#
#     y_origin = np.arange(len(traceSecondPart))
#     y_new = newIdxSecPart - len(trace1Interp)
#     traceInterp2 = np.linspace(0, len(traceSecondPart) - 1, y_new)
#     trace2Interp = np.interp(traceInterp2, y_origin, traceSecondPart)
#
#     interpArray = np.concatenate((trace1Interp, trace2Interp))
#     return interpArray
#
#
# class AlignStrainTraces:
#     def __init__(self, selected_source_data):
#         (self.avgMaxFrames, self.avgAvcFrames,
#          self.avgMaxPoints, self.avgAvcTimes) = None, None, None, None
#         self.idxAVCList, self.interpTracesDF = None, None
#         self.selected_source_data = selected_source_data
#         if self.selected_source_data == 'ST':
#             self.steTracesDF = dbTableNames['st_strainValuesInterp_T']
#             self.steSegmentalTracesDT = dbTableNames['st_segmentalStrainInterp_T']
#             self.query_call_view(sql_queries.select_ste_strain_traces_view())
#         elif self.selected_source_data == 'Philips':
#             self.steTracesDF = dbTableNames['phi_strainValuesInterp_T']
#             self.steSegmentalTracesDT = dbTableNames['phi_segmentalStrainInterp_T']
#             self.query_call_view(sql_queries.select_phi_strain_traces_view())
#
#         self.fix_spaces_df()
#         self.calculate_strain_reference_pts()
#         self.interpolate_traces()
#         self.save_interpolation_values()
#
#     def query_call_view(self, view_name_query):
#         dbConParams = dbConParameters
#         db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
#         db.connect()
#         db.execute_query(view_name_query)
#         self.tracesViewDF = db.df_query()
#         db.close_connection()
#
#     def fix_spaces_df(self):
#         tracesViewDF = self.tracesViewDF.copy()
#         tracesViewDF['cardiacView'] = tracesViewDF['cardiacView'].str.strip()
#         # self.interpTracesDF = pd.DataFrame()
#         columns_to_convert = ['strain_trace', 'timePoints', 'gls']
#         for column in columns_to_convert:
#             tracesViewDF[column] = tracesViewDF[column].str.strip('[]')
#             tracesViewDF[column] = tracesViewDF[column].apply(lambda x: [float(val) for val in x.split()])
#         # self.twoChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == '2ch'].copy()
#         # self.fourChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == '4ch'].copy()
#         # self.aplaxChTraces = tracesViewDF.loc[tracesViewDF['cardiacView'].str.lower() == 'aplax'].copy()
#
#
#     def interpolate_traces(self):
#         # two_ch_traces = self.twoChTraces
#         # four_ch_traces = self.fourChTraces
#         # aplax_ch_traces = self.aplaxChTraces
#         #
#         # carViewCols = [two_ch_traces, four_ch_traces, aplax_ch_traces]  # Info per cardiac view
#         # carViews = ['2ch', '4ch', 'Aplax']  # Cardiac view
#
#         #, avgMaxPoint, avgAvcFr, avgMaxFr
#         # for cvCol, carView, idxsAVC in zip(carViewCols, carViews, self.idxAVCList):
#         #                                 # , self.avgAvcTimes, self.avgMaxPoints,
#         #                                 #   self.avgAvcFrames, self.avgMaxFrames):
#         #
#         #     studyNbrs = cvCol.study_nr.tolist()
#         #     curve_segments = pd.Series(cvCol.curve_segment).str.strip().tolist()
#         #     segment_walls = pd.Series(cvCol.segment_wall).str.strip().tolist()
#         #     traces = cvCol.strain_trace.tolist()
#         #     ecgs = cvCol.ecg.tolist()
#         #     timePoints = cvCol.timePoints.tolist()
#         #     esFrames = cvCol.esFrame.tolist()
#         #     stGls = cvCol.gls.tolist()
#         #     newId_positions = cvCol.id_position.tolist()
#         #     newEd1Times = cvCol.ed1Time.tolist()
#         #     newEd1Frames = cvCol.ed1Frame.tolist()
#         #     newEd2Times = cvCol.ed2Time.tolist()
#         #     newEsTimes = cvCol.esTime.tolist()
#         #
#         #     newEsFrame, newEdFrame, newStudy_Nbr, newCardiacView, interp_ecg = [], [], [], [], []
#         #     newSegmentWall, x_interpStrainTraces, y_interpStrainTraces, gls_interpStrainTraces = [], [], [], []
#         #     newId_position, newEd1Frame, newEd1Time, newCurveSegment = [], [], [], []
#         #
#         #     for stCoord, idxAVC, time, esFrame, gls, new_Ed1Frame, new_ed1Time, \
#         #             study_Nbr, segmentWall, ecg, newIdposition, curveSegment, ed2_time,es_time in zip(traces, idxsAVC, timePoints, esFrames, stGls,
#         #                                                               newEd1Frames, newEd1Times,  studyNbrs, segment_walls,
#         #                                                                             ecgs, newId_positions, curve_segments,
#         #                                                                             newEd2Times, newEsTimes):
#         #         newCardiacView.append(carView)
#         #         newEdFrame.append(self.ptInterp_timePoints)
#         #         newEsFrame.append(self.max_es_frame)
#         #         newStudy_Nbr.append(study_Nbr)
#         #         newSegmentWall.append(segmentWall)
#         #         newCurveSegment.append(curveSegment)
#         #         newEd1Frame.append(new_Ed1Frame)
#         #         newEd1Time.append(new_ed1Time)
#         #         newId_position.append(int(newIdposition))
#         #         ecg_doubles = [float(value) for value in ecg.replace('[', '').replace(']', '').split()]
#         #
#         #
#         #         # X --> Interpolate STE points on X
#         #         x_interpStrainTraces.append(interpolate_array(stCoord, idxAVC, self.max_es_frame, self.ptInterp_timePoints))
#         #         # Y --> Interpolate time points on Y
#         #         y_interpStrainTraces.append(interpolate_array(time, esFrame, self.max_es_frame, self.ptInterp_timePoints))
#         #         # Interpolate ecg
#         #         interp_ecg.append(interpolate_array(ecg_doubles, idxAVC, self.max_es_frame, self.ptInterp_timePoints))
#         #         # Interpolate GLS
#         #         gls_interpStrainTraces.append(
#         #             interpolate_array(gls, idxAVC, self.max_es_frame, self.ptInterp_timePoints))  # TODO Fix to only be do it once per CV
#         #     strain_length = [len(row) for row in x_interpStrainTraces]
#         #     new_data = {
#         #         'study_Nbr': newStudy_Nbr,
#         #         'cardiacView': newCardiacView,
#         #         'segmentWall': newSegmentWall,
#         #         'ed1Frame': newEd1Frame,
#         #         'esFrame': newEsFrame,
#         #         'ed2Frame': newEdFrame,
#         #         'gls': gls_interpStrainTraces,
#         #         'ecg': interp_ecg,
#         #         'ed2Time': newEd2Times,
#         #         'esTime': newEsTimes,
#         #         'ed1Time': newEd1Time,
#         #         'timePoints': y_interpStrainTraces,
#         #         'curve_length': int(np.mean(strain_length)),
#         #         'curve_segment': newCurveSegment,
#         #         'id_position': newId_position,
#         #         'strainCoords': x_interpStrainTraces,
#         #     }
#         #     new_data_df = pd.DataFrame(new_data)
#         #     self.interpTracesDF = pd.concat([self.interpTracesDF, new_data_df], ignore_index=True)
#         #
#         # # Reset the index of the DataFrame
#         # self.interpTracesDF.reset_index(drop=True, inplace=True)
#
#     def calculate_strain_reference_pts(self):
#         # esFrame maximum
#         self.max_es_frame = np.max(self.tracesViewDF['esFrame'].values.tolist())
#         if self.selected_source_data == 'ST':
#             # time poimts maximum
#             self.ptInterp_timePoints = np.max(self.tracesViewDF['curve_length'].values.tolist())
#             pass
#         elif self.selected_source_data == 'Philips':
#             ste_segmental_interp_t = load_data_table_db('st_segmentalStrainInterp_T', False)
#             self.ptInterp_timePoints = np.max(ste_segmental_interp_t['curve_length'].values.tolist())
#             pass
#         # two_ch_traces = self.twoChTraces
#         # four_ch_traces = self.fourChTraces
#         # aplaxChTraces = self.aplaxChTraces
#
#         # max2chAVC = np.max(two_ch_traces['esFrame'].values.tolist())
#         # max4chAVC = np.max(four_ch_traces['esFrame'].values.tolist())
#         # maxAplaxAVC = np.max(aplaxChTraces['esFrame'].values.tolist())
#
#
#         # if self.selected_source_data == 'ST':
#         # max2chEd2Fr = np.max(two_ch_traces['ed2Frame'].values.tolist())
#         # max4chEd2Fr = np.max(four_ch_traces['ed2Frame'].values.tolist())
#         # maxAplaxEd2Fr = np.max(aplaxChTraces['ed2Frame'].values.tolist())
#         # elif self.selected_source_data == 'Philips':
#         #     steSegmentalInterpT = load_data_table_db('st_segmentalStrainInterp_T', False)
#         #     ste_curve_len = steSegmentalInterpT.iloc[0, 0]
#
#
#         # if self.selected_source_data == 'ST':
#         # elif self.selected_source_data == 'Philips':
#         # two_ch_lengths = [len(points) for points in two_ch_traces['timePoints']]
#         # four_ch_lengths = [len(points) for points in four_ch_traces['timePoints']]
#         # aplax_lengths = [len(points) for points in aplaxChTraces['timePoints']]
#         #
#         # # Get average list length
#         # avg2chNpts = round(np.average(two_ch_lengths))
#         # avg4chNpts = round(np.average(four_ch_lengths))
#         # avgAplaxNpts = round(np.average(aplax_lengths))
#         #
#         # # Nbr of points on AVC Index
#         # ptsIdx2chAvc = find_point_index(two_ch_traces['esTime'], two_ch_traces['timePoints'])
#         # ptsIdx4chAvc = find_point_index(four_ch_traces['esTime'], four_ch_traces['timePoints'])
#         # ptsIdxAplaxAvc = find_point_index(aplaxChTraces['esTime'], aplaxChTraces['timePoints'])
#         #
#         # # Nbr of average points
#         # avg2chAVCpts = np.average(ptsIdx2chAvc)
#         # avg4chAVCpts = np.average(ptsIdx4chAvc)
#         # avgAplaxAVCpts = np.average(ptsIdxAplaxAvc)
#         #
#         # # To interpolate x--> STE points,
#         # self.idxAVCList = [ptsIdx2chAvc, ptsIdx4chAvc, ptsIdxAplaxAvc]
#         # self.avgAvcTimes = [int(avg2chAVCpts), int(avg4chAVCpts), int(avgAplaxAVCpts)]
#         # self.avgMaxPoints = [int(avg2chNpts), int(avg4chNpts), int(avgAplaxNpts)]
#         #
#         # # To interpolate y-->  time points
#         # self.avgAvcFrames = [int(max2chAVC), int(max4chAVC), int(maxAplaxAVC)]
#         # self.avgMaxFrames = [int(max2chEd2Fr), int(max4chEd2Fr), int(maxAplaxEd2Fr)]
#
#     def save_interpolation_values(self):
#         study_Nbr = self.interpTracesDF['study_Nbr'].tolist()
#         cardiacView = self.interpTracesDF['cardiacView'].tolist()
#         segmentName = self.interpTracesDF['segmentWall'].tolist()
#         curve_segment = self.interpTracesDF['curve_segment'].tolist()
#         ed1Frames = self.interpTracesDF['ed1Frame'].tolist()
#         esFrames = self.interpTracesDF['esFrame'].tolist()
#         ed2Frames = self.interpTracesDF['ed2Frame'].tolist()
#         glss = self.interpTracesDF['gls']
#         ecgs = self.interpTracesDF['ecg']
#         ed1Times = self.interpTracesDF['ed1Time']
#         ed2Times = self.interpTracesDF['ed2Time']
#         esTimes = self.interpTracesDF['esTime']
#         timePoints = self.interpTracesDF['timePoints']
#         idPosition = self.interpTracesDF['id_position']
#         curve_len = self.interpTracesDF['curve_length']
#
#         dbConParams = dbConParameters
#         db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
#         db.connect()
#         db.delete_query(self.steTracesDF)
#
#         for (study, cview, segment, ed1Frame, esFrame, ed2Frame, gls, ecg, ed2Time,
#              esTime, ed1Time, timePts) in zip(study_Nbr, cardiacView, segmentName, ed1Frames, esFrames, ed2Frames,
#                                               glss, ecgs, ed2Times, esTimes, ed1Times, timePoints):
#             rows = []
#             rows = (study, cview, segment, ed1Frame, esFrame, ed2Frame, str(gls), str(ecg), ed2Time, esTime, ed1Time,
#                     str(timePts), 'BL', 0)
#             db.insert_values(self.steTracesDF, rows)
#
#
#         db.delete_query(self.steSegmentalTracesDT)
#         strainCoords = self.interpTracesDF['strainCoords']
#         for (study, segment, curvelen, curveseg, idposition, straintrace, cview) \
#                 in zip(study_Nbr, segmentName, curve_len, curve_segment, idPosition, strainCoords, cardiacView):
#             rows = []
#             rows = (study, segment, curvelen, curveseg, idposition, str(straintrace), cview,'BL', 0)
#             db.insert_values(self.steSegmentalTracesDT, rows)
#             pass
#         db.close_connection()
