import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from dash_viewer.classification.deep_learning.n_beats_classifier import NBeats_classifier
from dash_viewer.classification.machine_learning.best_params_classifiers.bestClassSteEchoParam_viewer import \
    BestClassSteEchoParamViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.best_echoparms_class_viewer import \
    BestEchoParamsViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.best_vac_class_viewer import BestParamClassVacViewer
# from dash_viewer.classification.machine_learning.classifiers.gb_class_viewer import GBViewer
from dash_viewer.classification.machine_learning.classifiers.knn_class_viewer import KNNViewer
from dash_viewer.classification.machine_learning.classifiers.dt_class_viewer import DTViewer
from dash_viewer.classification.machine_learning.classifiers.rf_class_viewer import RFClassViewer
from dash_viewer.classification.machine_learning.classifiers.svm_class_viewer import SVMClassViewer
# from dash_viewer.classification.machine_learning.classifiers.xgboost_class_viewer import XGBClassViewer
# from dash_viewer.classification.machine_learning.classifiers.catBoost_class_viewer import CatBoostClassViewer
from dash_viewer.patient_plot_viewer import PatientTracesViewer
from features.navbar import create_navbar
from curve_plotter_viewer import TracesViewer
from remodeling_viewer import RemodelingViewer
from dash_viewer.classification.machine_learning.best_params_classifiers.best_knn_class_viewer import BestParamClassViewer
from dash_viewer.data_imputation.data_best_imputation_viewer import DataImputationKBest
from dash_viewer.data_imputation.stTraces_imputation_viewer import DataImputationST
from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import DimensionalityRedViewer
from merge_remod_traces_viewer import StrainRemodelingViewer

app = dash.Dash(external_stylesheets=[dbc.themes.SPACELAB],
                suppress_callback_exceptions=True)  # Changed theme to CERULEAN

navbar = create_navbar()

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content'),
    html.Div(id='content-div'),
    html.Div(id='hidden-div', style={'display': 'none'})
])


# tab0 = WelcomeViewer(app)
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
    elif pathname == '/tab-best_class':  # Best Classification Algorithm parameters
        tab = BestParamClassViewer()
        content = tab.get_content()
    elif pathname == '/tab-best_vac_class':  # Best Classification Algorithm parameters
        tab = BestParamClassVacViewer()
        content = tab.get_content()
    elif pathname == '/tab-bestEchoParam_class':  # KNN
        tab = BestEchoParamsViewer()
        content = tab.get_content()
    elif pathname == '/tab-bestSteEcho_class':  # KNN
        tab = BestClassSteEchoParamViewer()
        content = tab.get_content()

    elif pathname == '/tab-KNN_class':  # KNN
        tab = KNNViewer()
        content = tab.get_content()
    elif pathname == '/tab-DT_class':  # KNN
        tab = DTViewer()
        content = tab.get_content()
    elif pathname == '/tab-RF_class':  # Random Forest
        tab = RFClassViewer()
        content = tab.get_content()
    # elif pathname == '/tab-GB_class':  # Gradient Boosting
    #     tab = GBViewer()
    #     content = tab.get_content()
    elif pathname == '/tab-SVM_class':  # SVM
        tab = SVMClassViewer()
        content = tab.get_content()
    # elif pathname == '/tab-XGB_class': # XGBoost
    #     tab = XGBClassViewer()
    #     content = tab.get_content()
    # elif pathname == '/tab-CatBoost_class':  # CatBoost
    #     tab = CatBoostClassViewer()
    #     content = tab.get_content()
    elif pathname == '/tab-autoencoder_class':
        tab = NBeats_classifier()
        content = tab.get_content()
    else:
        return dash.no_update

    return content


if __name__ == '__main__':
    app.run_server(debug=True)
