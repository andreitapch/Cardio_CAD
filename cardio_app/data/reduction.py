import json

import numpy as np

from cardio_app import settings
from cardio_app.data import db, traces
from cardio_app.services import reduction

TRACES, PCA_RESULTS, PCA_PARAMS, ENCODED = ("calculated_reconstructedTraces", "calculated_pca_results",
                                            "calculated_pca_params", "calculated_dr_encoded_data")


def reconstructed_type(method, concat, components_key, imputed):
    return f"{method}-{concat}-{components_key}-{int(imputed)}"


def save(method, concat, source, items, result, imputed_studies):
    source_key = traces.SOURCE_KEYS.get(source, source)
    ks = sorted(set(int(k) for k in result.k.values()))
    components_key = ks[0] if len(ks) == 1 else "_".join(map(str, ks))
    n_components = max(ks)
    flag = lambda study: int(study in imputed_studies)
    types = [reconstructed_type(method, concat, components_key, i) for i in (0, 1)]
    frame = reduction.reconstructed_frame(items, result)
    trace_rows = [(int(r.study_nr), r.curve_segment, int(r.id_position), traces.format_trace(r.trace), settings.STUDY_TYPE,
                   reconstructed_type(method, concat, components_key, flag(r.study_nr)), float(n_components), flag(r.study_nr), source_key)
                  for r in frame.itertuples()]
    studies, codes = reduction.stitched_codes(items, result)
    code_rows = [(int(study), json.dumps(np.asarray(row).tolist()), reconstructed_type(method, concat, components_key, flag(study)),
                  float(n_components), flag(study), source_key) for study, row in zip(studies, codes)]
    batches = [(f"DELETE FROM {TRACES} WHERE reconstructed_type = ? AND data_source = ?", [(t, source_key) for t in types]),
               (f"INSERT INTO {TRACES} (study_nr, curve_segment, id_position, strain_trace, studyType, reconstructed_type, "
                f"n_components, isImputed, data_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", trace_rows)]
    if method == "PCA":
        param_rows = []
        for block in items:
            model = result.models[block.name]
            label = concat if block.name == reduction.FULL else f"{concat}-{block.name}"
            param_rows.append((json.dumps(model["components"].tolist()), json.dumps(model["explained_variance"].tolist()),
                               json.dumps(model["explained_variance_ratio"].tolist()), json.dumps(model["mean"].tolist()),
                               int(result.k[block.name]), int(model["components"].shape[1]), json.dumps(model["singular_values"].tolist()),
                               label, float(n_components), 0, source_key))
        batches += [(f"DELETE FROM {PCA_RESULTS} WHERE reconstructed_type = ? AND data_source = ?", [(t, source_key) for t in types]),
                    (f"INSERT INTO {PCA_RESULTS} (study_nr, pca_results, reconstructed_type, n_components, isImputed, data_source) "
                     f"VALUES (?, ?, ?, ?, ?, ?)", code_rows),
                    (f"DELETE FROM {PCA_PARAMS} WHERE reconstructed_type = ? AND data_source = ? AND n_components_ = ?",
                     [(row[7], source_key, float(n_components)) for row in param_rows]),
                    (f"INSERT INTO {PCA_PARAMS} (components_, explained_variance_, explained_variance_ratio_, mean_, n_components, "
                     f"n_features, singular_values_, reconstructed_type, n_components_, isImputed, data_source) "
                     f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", param_rows)]
    else:
        batches += [(f"DELETE FROM {ENCODED} WHERE reconstructed_type = ? AND data_source = ?", [(t, source_key) for t in types]),
                    (f"INSERT INTO {ENCODED} (study_nr, encoded_data, reconstructed_type, encoding, isImputed, data_source) "
                     f"VALUES (?, ?, ?, ?, ?, ?)", code_rows)]
    return db.execute_many(batches)
