from cardio_app.services.classification.clinical import split_important_features
from cardio_app.services.classification.data import (get_concat_18_segments, get_echoParams_data, get_feature_components,
                                                     load_data_traces, load_loadings, load_queries_table_db, select_data_type)
from cardio_app.services.classification.engine import (DEFAULT_REDUCER_COMPONENTS, convert_echoParam_values,
                                                       default_scaling_for, find_best_class_params, load_selected_features,
                                                       preprocess_data)

SCALING = {"auto": None, "on": True, "off": False}


def load_ste_echo(input_type, imputed, data_source, encoding):
    filter_by = select_data_type(input_type, encoding)
    traces = load_queries_table_db("pca_results_view", "data_source", "in", [data_source])
    filtered = traces[traces["reconstructed_type"].str.contains(filter_by, na=False)]
    echo, _ = get_echoParams_data(filtered, data_source, None)
    x, _, info, _ = preprocess_data(convert_echoParam_values(echo), True, fit_preprocessing=False)
    _, selected = load_selected_features()
    features = x.loc[:, [column for column in x.columns if column in selected]].copy()
    features.insert(0, "study_nr", info["study_nr"])
    original = traces[traces["reconstructed_type"] == f"{filter_by}-0"]
    imputed_frame = traces[traces["reconstructed_type"] == f"{filter_by}-1"] if imputed else None

    def prepare(frame, flag):
        if frame is None:
            return None, None
        if input_type == "reconstructed_traces_vac":
            concat, column = get_concat_18_segments(frame)
        else:
            concat, column = get_feature_components(frame, data_source, filter_by, "encoded_data",
                                                    "calculated_dr_encoded_data_T", flag)
        return concat.merge(features, on="study_nr", how="left"), column

    orig, orig_column = prepare(original, False)
    imp, imp_column = prepare(imputed_frame, True)
    return orig, orig_column, imp, imp_column, None


def load(config):
    if config["kind"] == "ste_echo":
        return load_ste_echo(config["input_type"], config["imputed"], config["data_source"], config["encoding"])
    return load_data_traces(config["input_type"], config["imputed"], config["unbalanced_type"], config["class_type"],
                            config["data_source"], config["n_components"], config["add_gls"], config["add_echo"],
                            None, config["class_input"])


def run(config):
    orig, orig_column, imp, imp_column, features = load(config)
    class_input = config.get("class_input")
    scale = SCALING.get(config.get("scaling_mode", "auto"))
    if scale is None:
        scale = default_scaling_for(class_input if class_input else config["class_type"])
    reducer = config.get("reducer")
    reducer = None if reducer in (None, "none") else reducer
    (_, best, inner, metrics, cm, shap_values, shap_input, important, shap_df) = find_best_class_params(
        orig, orig_column, imp, imp_column, config["unbalanced_type"], config["class_type"], features,
        scale_features=scale, class_input=class_input, reducer=reducer,
        reducer_components=config.get("reducer_components", DEFAULT_REDUCER_COMPONENTS), autosave=True)
    result = {"config": dict(config, scale_features=scale), "best_params": best, "inner_score": inner, "metrics": metrics,
              "cm": cm, "shap_values": shap_values, "shap_input_df": shap_input, "important_features": important or [],
              "shap_df": shap_df, "n_rows": len(orig)}
    if best and important and config["kind"] == "classify":
        pca_features, clinical = split_important_features(important)
        result["pca_features"], result["clinical_features"] = pca_features, clinical
        if pca_features:
            result["loadings"] = load_loadings(config["data_source"], config["input_type"], config["imputed"],
                                               config["n_components"], "calculated_pca_params_T")
    return result
