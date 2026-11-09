from dash import dcc
import dash_bootstrap_components as dbc_components

import dash_bootstrap_components as dbc
from dash import html


def create_navbar():
    navbar = dbc.Navbar(
        [
            html.A(
                dbc.Row(
                    [
                        # dbc.Col(html.Img(src="/static/img/brand.svg", height="30px")),
                        dbc.Col(dbc.NavbarBrand("CAD", className="ml-2")),
                    ],
                    align="center",
                ),
                href="#",
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav(
                    [
                        # dbc.NavItem(dbc.NavLink("Welcome", href='/tab-WELCOME')),
                        dbc.NavItem(dbc.NavLink("Remodeling Data", href='/tab-Remodeling')),
                        dbc.NavItem(dbc.NavLink("Remodeling with Strain Traces", href='/tab-dfMerger')),
                        dbc.NavItem(dbc.NavLink("Temporal Alignment", href='/tab-TempAlignment')),
                        dbc.NavItem(dbc.NavLink("Patient Viewer", href='/tab-PatViewer')),
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem("Find the best K", href='/tab-bestK'),
                                dbc.DropdownMenuItem("Knn Imputation ST", href='/tab-missingData-ST'),
                                dbc.DropdownMenuItem("Knn Imputation Params", href='/tab-missingData-Bio'),
                            ],
                            nav=True,
                            in_navbar=True,
                            label="Data Imputation",
                        ),
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem("PCA", href='/tab-PCA'),
                                dbc.DropdownMenuItem("Autoencoder", href='/tab-AutoencoderDm'),#TODO
                                dbc.DropdownMenuItem("Comparison", href='/tab-Comparison'),#TODO
                            ],
                            nav=True,
                            in_navbar=True,
                            label="Dimensionality Reduction",
                        ),
                        dbc.DropdownMenu(
                            [
                                dbc.DropdownMenuItem("Find best class Params: PCA", href='/tab-best_class'),
                                dbc.DropdownMenuItem("Find best class Params: VAC", href='/tab-best_vac_class'),
                                dbc.DropdownMenuItem("Find Echo Params: ", href='/tab-bestEchoParam_class'),
                                dbc.DropdownMenuItem("Classify Ste & Echo Params: ", href='/tab-bestSteEcho_class'),
                                dbc.DropdownMenuItem("Compare Classifiers", href='/tab-class_comparison'),
                                # dbc.DropdownMenuItem("KNN classifier", href='/tab-KNN_class'),
                                # dbc.DropdownMenuItem("Decision Tree  classifier", href='/tab-DT_class'),
                                # dbc.DropdownMenuItem("RF classifier", href='/tab-RF_class'),
                                # dbc.DropdownMenuItem("SVM classifier", href='/tab-SVM_class'),
                                # dbc.DropdownMenuItem("gb classifier", href='/tab-GB_class'),
                                # dbc.DropdownMenuItem("Catboost classifier", href='/tab-CatBoost_class'),
                                # dbc.DropdownMenuItem("xgboost classifier", href='/tab-XGB_class'),
                                # dbc.DropdownMenuItem("AutoencoderDm classifier", href='/tab-autoencoder_class'),
                            ],
                            nav=True,
                            in_navbar=True,
                            label="Classifier",
                        ),
                    ],
                    pills=True,
                    className="ml-auto",
                ),
                id="navbar-collapse",
                navbar=True,
            ),
        ],
        color="primary",
        dark=True,
    )
    return navbar
