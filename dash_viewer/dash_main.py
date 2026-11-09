import os

# MKL picks its OpenMP runtime at load time. Left to itself it loads conda's
# libomp.dll, which aborts the whole process ("OMP: Error #15") once torch's
# libiomp5md.dll is initialised - matplotlib rendering a SHAP plot was enough to
# kill the server mid-request. TBB keeps MKL threaded without any OpenMP runtime.
# Must be set before numpy/MKL is imported, so it stays at the top of the file.
os.environ.setdefault('MKL_THREADING_LAYER', 'TBB')

# Torch must then load before any MKL-linked import (scipy, sklearn, shap):
# whichever comes first claims libiomp5md.dll process-wide, and if conda's MKL
# wins, torch's shm.dll fails to load with WinError 127.
try:
    import torch  # noqa: F401
except Exception:  # torch is optional - the LLM tab degrades without it
    pass

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# from dash_viewer.classification.deep_learning.n_beats_classifier import NBeats_classifier
from dash_viewer.classification.machine_learning.best_params_classifiers.bestClassSteEchoParam_viewer import \
    BestClassSteEchoParamViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.featSelection_params_class_viewer import \
    BestEchoParamsViewer
# from dash_viewer.classification.machine_learning.best_params_classifiers.best_vac_class_viewer import BestParamClassVacViewer
# from dash_viewer.classification.machine_learning.classifiers.knn_class_viewer import KNNViewer
# from dash_viewer.classification.machine_learning.classifiers.dt_class_viewer import DTViewer
# from dash_viewer.classification.machine_learning.classifiers.rf_class_viewer import RFClassViewer
# from dash_viewer.classification.machine_learning.classifiers.svm_class_viewer import SVMClassViewer
from dash_viewer.patient_plot_viewer import PatientTracesViewer
from features.navbar import create_navbar
from curve_plotter_viewer import TracesViewer
from remodeling_viewer import RemodelingViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.best_knn_class_viewer import BestParamClassViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.class_comparison_viewer import \
    ClassComparisonViewer
from dash_viewer.data_imputation.data_best_imputation_viewer import DataImputationKBest
from dash_viewer.data_imputation.stTraces_imputation_viewer import DataImputationST
from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import DimensionalityRedViewer
from dash_viewer.dimensionality_reduction.comparison_viewer import ComparisonViewer
from merge_remod_traces_viewer import StrainRemodelingViewer

app = dash.Dash(external_stylesheets=[dbc.themes.SPACELAB],
                suppress_callback_exceptions=True)

navbar = create_navbar()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    html.Div(id='content-div'),
    html.Div(id='hidden-div', style={'display': 'none'})
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    return dcc.Loading(
        id="loading",
        type="circle",
        fullscreen=False,
        children=[]
    )


@app.callback(Output('loading', 'children'),
              Input('url', 'pathname'))
def load_content(pathname):
    if pathname == '/tab-Remodeling':
        tab = RemodelingViewer()
        content = tab.get_content()
    elif pathname == '/tab-dfMerger':
        tab = StrainRemodelingViewer()
        content = tab.get_content()
    elif pathname == '/tab-TempAlignment':
        tab = TracesViewer()
        content = tab.get_content()
    elif pathname == '/tab-PatViewer':
        tab = PatientTracesViewer()
        content = tab.get_content()
    elif pathname == '/tab-bestK':
        tab = DataImputationKBest()
        content = tab.get_content()
    elif pathname == '/tab-missingData-ST':
        tab = DataImputationST()
        content = tab.get_content()
    # elif pathname == '/tab-missingData-Bio':#TODO
    #     tab = DataImputationParams()
    #     content = tab.get_content()
    elif pathname == '/tab-PCA':
        tab = DimensionalityRedViewer(pathname)
        content = tab.get_content()
    elif pathname == '/tab-AutoencoderDm':
        tab = DimensionalityRedViewer(pathname)
        content = tab.get_content()
    elif pathname == '/tab-Comparison':
        tab = ComparisonViewer(pathname)
        content = tab.get_content()
    elif pathname == '/tab-best_class':  # Best Classification Algorithm parameters
        tab = BestParamClassViewer()
        content = tab.get_content()
    # elif pathname == '/tab-best_vac_class':  # Best Classification Algorithm parameters
    #     tab = BestParamClassVacViewer()
    #     content = tab.get_content()
    elif pathname == '/tab-bestEchoParam_class':  # KNN
        tab = BestEchoParamsViewer()
        content = tab.get_content()
    elif pathname == '/tab-bestSteEcho_class':  # KNN
        tab = BestClassSteEchoParamViewer()
        content = tab.get_content()
    elif pathname == '/tab-class_comparison':  # All classifiers, same data, same folds
        tab = ClassComparisonViewer(pathname)
        content = tab.get_content()

    # elif pathname == '/tab-KNN_class':  # KNN
    #     tab = KNNViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-DT_class':  # KNN
    #     tab = DTViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-RF_class':  # Random Forest
    #     tab = RFClassViewer()
    #     content = tab.get_content()
    # # elif pathname == '/tab-GB_class':  # Gradient Boosting
    # #     tab = GBViewer()
    # #     content = tab.get_content()
    # elif pathname == '/tab-SVM_class':  # SVM
    #     tab = SVMClassViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-XGB_class': # XGBoost
    #     tab = XGBClassViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-CatBoost_class':  # CatBoost
    #     tab = CatBoostClassViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-autoencoder_class':
    #     tab = NBeats_classifier()
    #     content = tab.get_content()
    else:
        return dash.no_update

    return content


if __name__ == '__main__':
    # Reloading OFF, both mechanisms, because they are separate:
    #   dev_tools_hot_reload=False stops Dash's browser-side reload polling;
    #   use_reloader=False (passed through to Flask/werkzeug) stops the process
    #   itself being restarted when a .py file changes.
    # debug=True alone enables the second one, so disabling only the first is not
    # enough. A restart kills any classification sweep in flight - hours of
    # nested CV lost to an unrelated edit, with the browser quietly reloading and
    # the leaderboard going blank. The debugger and error pages are unaffected;
    # restart manually after changing code.
    app.run(debug=True, use_reloader=False, dev_tools_hot_reload=False)
