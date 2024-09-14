# from  pattern_recognition.utils.parameters_dics import segment_name_dict
#
#
# def map_labels_to_names(labels):
#     names = []
#     for label in labels:
#         name = segment_name_dict[str(label)]['description']
#         names.append(name)
#     return names
#
#
# class CurveLabeler:
#     def __init__(self, curves):
#         self.curve_labelled = []
#         labels = curves.label_sorted
#         map_labels_to_names(labels)
#
#     def label_curves(self, curves):
#         for i in range(len(curves)):
#             for j in range(len(curves[i])):
#                 labels = curves.label_sorted[i]
#                 names = map_labels_to_names(labels)
#                 self.curve_labelled[i][j] = (curves[i][j], names)
