import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from pattern_recognition.utils.plot_dics import segment_color_dict

segment_labels = [
    "apAnt", "apAntSept", "apInf", "apLat", "apPost", "apSept",
    "basAnt", "basAntSept", "basInf", "basLat", "basPost", "basSept",
    "midAnt", "midAntSept", "midInf", "midLat", "midPost", "midSept"
]

def detect_post_systolic_peak(segment_curve, es_frame):
    peaks, _ = find_peaks(segment_curve)
    for p in peaks:
        if p > es_frame:
            return True
    return False


def get_segment_contributions(reshaped_loading_matrix):
    contributions = [np.sum(np.abs(seg)) for seg in reshaped_loading_matrix]
    return contributions


def split_important_features(important_features):
    """
    Split important features into PCA components and clinical features.
    """
    pca_features = [f for f in important_features if f.startswith('pca_results_')]
    clinical_features = [f for f in important_features if not f.startswith('pca_results_')]
    return pca_features, clinical_features


# def plot_pca_loading_segments(pca_loading_vector, component_idx, n_segments, n_timepoints):
#     """
#     Plot PCA loading curves per LV segment with anatomical labels.
#     """
#     es_frame = 54
#     pca_loading_vector = np.array(pca_loading_vector)
#     reshaped = np.split(pca_loading_vector, n_segments)
#
#     fig = go.Figure()
#     for idx in range(n_segments):
#         segment_label = segment_labels[idx]
#         fig.add_trace(go.Scatter(
#             y=reshaped[idx],
#             mode='lines',
#             name=segment_label,
#             line=dict(color=segment_color_dict[segment_label])
#         ))
#
#     fig.add_vline(
#         x=es_frame,
#         line_width=2,
#         line_dash="dot",
#         line_color="black",
#         annotation_text="End-Systole",
#         annotation_position="top right"
#     )
#
#
#     fig.update_layout(
#         title=f"PCA Component {component_idx} Loadings by Segment",
#         title_x=0.5,
#         xaxis_title="Timepoints",
#         yaxis_title="Loading Weight",
#         height=600,
#         template="plotly_white"
#     )
#
#     return fig


def analyze_deformation_patterns(segment_curve, es_frame):
    peaks, _ = find_peaks(segment_curve)
    is_double_peak = len(peaks) >= 2
    is_flattened = np.std(segment_curve) < 0.02
    early_peak = any(p < es_frame - 3 for p in peaks)
    return {
        "early_peak": early_peak,
        "double_peak": is_double_peak,
        "flattened": is_flattened,
        "peak_positions": peaks.tolist()
    }


def loading_segments(pca_loading_vector, n_segments=18, es_frame=54, top_n=3):
    reshaped = np.split(np.array(pca_loading_vector), n_segments)
    contribution_scores = get_segment_contributions(reshaped)
    top_indices = set(np.argsort(contribution_scores)[-top_n:])
    rows, curves = [], []
    for idx in range(n_segments):
        curve = reshaped[idx]
        deformation = analyze_deformation_patterns(curve, es_frame)
        rows.append({"Segment": segment_labels[idx], "Contribution": contribution_scores[idx],
                     "Early Peak": deformation["early_peak"], "Double Peak": deformation["double_peak"],
                     "Flattened": deformation["flattened"],
                     "Post-Systolic Peak": detect_post_systolic_peak(curve, es_frame)})
        curves.append((segment_labels[idx], curve, idx in top_indices))
    return curves, pd.DataFrame(rows).sort_values("Contribution", ascending=False)
