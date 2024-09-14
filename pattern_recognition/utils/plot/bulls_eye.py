from utils.SmoothAHAplot.src.aha.aha import AHA


# def plot_missing_bulls_eye(missing_segments):
#
#     segments_18 = ['Basal Anterior', 'Basal Anteroseptal', 'Basal Inferoseptal',
#                    'Basal Inferior', 'Basal Inferolateral', 'Basal Anterolateral',
#                    'Mid Anterior', 'Mid Anteroseptal', 'Mid Inferoseptal',
#                    'Mid Inferior', 'Mid Inferolateral', 'Mid Anterolateral',
#                    'Apical Anterior', 'Apical Anteroseptal', 'Apical Inferoseptal',
#                    'Apical Inferior', 'Apical Inferolateral', 'Apical Anterolateral']
#     exp_strain_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
#     n_seg = len(exp_strain_data)
#     strain_dict = {k: v for (k, v) in zip(segments_18, exp_strain_data)}
#
#     aha = AHA(strain_dict, 'Strain', None)
#     fig = aha.bullseye_smooth(missing_segments)
#     fig.show()


