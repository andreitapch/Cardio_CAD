from dash import html

class ClinicalMetricsViewer:
    def __init__(self, selected_source_data=None, concat_traces_by=None, studyType=None):
        self.selected_source_data = selected_source_data
        self.concat_traces_by = concat_traces_by
        self.studyType = studyType

    def get_content(self):
        # Call the function from data_processing module

        return html.Div([
            html.H3('Tab 1'),
            html.P('Hello, World!')
        ])