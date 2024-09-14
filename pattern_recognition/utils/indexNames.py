class IndexNames:
    def __init__(self):
        self.ed2Time, self.study_nr, self.TracesStudy_Nbr, = None, None, None
        self.nrFrames, self.pointQty, self.timePoints, self.strainCoords = None, None, None, None
        self.segmentName, self.ed1Time, self.esTime, self.stValuesDf, self.ecg = None, None, None, None, None
        self.ed2Frame, self.esFrame, self.ed1Frame, self.cardiacView, self.globalSt = None, None, None, None, None

    def indexNames_stTracesDf(self, column_stTracesDf):
        self.TracesStudy_Nbr = column_stTracesDf[0]
        self.segmentName = column_stTracesDf[1]
        self.strainCoords = column_stTracesDf[2]
        self.timePoints = column_stTracesDf[3]
        self.pointQty = column_stTracesDf[4]
        self.nrFrames = column_stTracesDf[5]

    def indexNames_stValuesDf(self, column_stValuesDf):
        self.study_nr = column_stValuesDf[0]
        self.cardiacView = column_stValuesDf[1]
        self.ed1Frame = column_stValuesDf[2]
        self.esFrame = column_stValuesDf[3]
        self.ed2Frame = column_stValuesDf[4]
        self.globalSt = column_stValuesDf[5]
        self.ecg = column_stValuesDf[6]
        self.ed2Time = column_stValuesDf[7]
        self.esTime = column_stValuesDf[8]
        self.ed1Time = column_stValuesDf[9]
