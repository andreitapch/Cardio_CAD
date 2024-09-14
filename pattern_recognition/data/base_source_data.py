class BaseSourceData:
    def __init__(self, studyType, concat_traces_by):
        self.studyType = studyType
        self.concat_traces_by = concat_traces_by

    def process_data(self):
        raise NotImplementedError