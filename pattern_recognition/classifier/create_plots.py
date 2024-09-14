import os
import plotly.offline as pyo
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

# from pattern_recognition.utils.parameters_dics import segment_name_dict as segment_dict
from  pattern_recognition.utils.parameters_dics import segment_color_dict as color_dict
from collections import defaultdict
import plotly.graph_objects as go


# segment_description_dict = {info['segment']: info['description']
#                             for info in segment_dict.values()}
################################################################################
##                                 PLOT PCA POINTS                            ##
################################################################################
def plot_pca(pc,name):
    fig = go.Figure()

    for i in range(pc.shape[0]):
        fig.add_trace(go.Scatter(
            x=list(range(pc.shape[1])),
            y=pc[i, :],
            mode='lines',
            name=f'Principal Component {i + 1}'
        ))

    fig.update_layout(
        title='PCA Line Plot ' + name,
        xaxis_title='Index',
        yaxis_title='Value',
        legend_title='Principal Components'
    )

    fig.show()



def plot_principal_components(principal_components):
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'yellow']

    for i in range(principal_components.shape[1]):
        fig.add_trace(go.Scatter(
            x=list(range(principal_components.shape[0])),
            y=principal_components[:, i],
            mode='markers',
            marker=dict(size=5, color=colors[i % len(colors)]),
            name=f'Principal Component {i + 1}'
        ))

    fig.update_layout(title='Principal Components',
                      xaxis_title='Index',
                      yaxis_title='Value')
    fig.show()
################################################################################
##                               PLOT STRAIN CURVES                           ##
################################################################################
# def get_segment_curves(curves, labels, sub_length):
#     segment_curves = defaultdict(lambda: defaultdict(list))
#
#     for i, sub_list in enumerate(curves):
#         for j, sub_sublist in enumerate(sub_list):
#             for k, row in enumerate(sub_sublist):
#                 segment = segment_dict[str(labels[i][j][k][0])]['segment']
#                 if i < sub_length['control']:
#                     segment_curves['control'][segment].append((row, f'{segment}'))
#                 else:
#                     segment_curves['abnormal'][segment].append((row, f'{segment}'))
#     return segment_curves


def create_subplot(segment_curves,file_name):
    max_curve_length = max(len(curve) for segment in segment_curves['control']
                           for curve, _ in segment_curves['control'][segment])

    specs = [[{}, {}] for _ in range(len(segment_curves['control']))]
    fig = make_subplots(rows=len(segment_curves['control']), cols=2,
                        specs=specs, subplot_titles=([segment for segment in
                                                      segment_curves['control'].keys() for _ in range(2)]))
    fig.update_layout(showlegend=False, title_text=file_name + ' Control - Abnormal', title_x=0.5)

    fig.update_layout(height=10 * max_curve_length)
    return fig


def add_traces_to_subplot(fig, segment_curves, subject_type, col):
    for i, (segment, curves) in enumerate(segment_curves[subject_type].items()):
        for curve, name in curves:
            fig.add_trace(go.Scatter(y=curve, mode='lines', name=f'{segment_description_dict[segment]}',
                                     line=dict(color=color_dict[segment]), showlegend=False), row=i + 1, col=col)
    return fig


# def plot_curves(curves, labels, ev_times, sub_length, file_name):
#     segment_curves = get_segment_curves(curves, labels, sub_length)
#     fig = create_subplot(segment_curves,file_name)
#     fig = add_traces_to_subplot(fig, segment_curves, 'control', 1)
#     fig = add_traces_to_subplot(fig, segment_curves, 'abnormal', 2)
#     # pyo.iplot(fig)
#
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     pyo.iplot(fig, filename=os.path.join(current_directory,'plots', file_name + '.html'))
#     plt.show()
