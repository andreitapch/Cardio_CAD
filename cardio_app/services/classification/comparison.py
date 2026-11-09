import json
import threading
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from cardio_app import settings
from cardio_app.services.classification.data import load_data_traces
from cardio_app.services.classification.engine import (
    evaluate_classifier, default_scaling_for, DEFAULT_N_REPEATS,
    DEFAULT_REDUCER_COMPONENTS)
from cardio_app.services.classification.model_params import CLASSIFIER_LABELS, COMPARISON_DEFAULT_MODELS

# Same per-session pattern as the dimensionality-reduction comparison tab: a
# sweep holds pooled predictions and SHAP frames for every model, far too much
# to round-trip through dcc.Store.

DATA_SOURCE_MAPPING = {
    'Speqle': 'speqle_segmentalStrainInterp_T',
    'GE': 'st_segmentalStrainInterp_T',
    'Philips': 'phi_segmentalStrainInterp_T',
    'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T'],
}

# <repo>/dash_viewer/classification/machine_learning/best_params_classifiers/<this file>
REPORTS_DIR = settings.REPORTS

# Every model is scored on the same folds; that only holds if the seed is fixed
# here and never taken from the UI.
COMPARISON_RANDOM_STATE = 42
DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 5
DEFAULT_SEARCH_BUDGET = 30
# Searching the sampler multiplies the grid by roughly the number of sampler
# options (none / SMOTE / ADASYN / class-weight).
SAMPLER_SEARCH_FACTOR = 4

IMBALANCE_OPTIONS = [
    {'label': 'None', 'value': 'None'},
    {'label': 'SMOTE', 'value': 'SMOTE'},
    {'label': 'ADASYN', 'value': 'ADASYN'},
    {'label': 'Balanced', 'value': 'Balanced'},
    {'label': 'Search per fold (recommended)', 'value': 'search'},
]

REDUCER_OPTIONS = [
    {'label': 'None', 'value': 'none'},
    {'label': 'PCA', 'value': 'pca'},
    {'label': 'UMAP', 'value': 'umap'},
]

# Measured on a 250x72 matrix. Only UMAP is slow enough to be worth warning
# about; a PCA fit on this cohort is milliseconds.
UMAP_SECONDS_PER_FIT = 25


def _label_for(class_type):
    return CLASSIFIER_LABELS.get(class_type, class_type)


def _fmt(value, digits=3):
    if value is None:
        return 'n/a'
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 'n/a'
    if np.isnan(value):
        return 'n/a'
    return f'{value:.{digits}f}'


def _fmt_ci(bounds):
    if not bounds:
        return ''
    try:
        low, high = bounds
    except (TypeError, ValueError):
        return ''
    return f'[{_fmt(low)}, {_fmt(high)}]'


def _pr_auc(result):
    """Pooled PR-AUC: one curve over all held-out predictions, not a fold mean.

    Averaging per-fold PR-AUC over folds that each hold a handful of positives
    is unstable; the pooled figure is what the leaderboard ranks on.
    """
    pooled = result.get('pooled_metrics') or {}
    value = pooled.get('PR-AUC_global')
    if value is None:
        value = (result.get('metrics') or {}).get('PR-AUC')
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def _baseline(result):
    pooled = result.get('pooled_metrics') or {}
    value = pooled.get('PR-AUC_baseline', pooled.get('Prevalence'))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def _pr_auc_ci(result):
    ci = result.get('ci') or {}
    return ci.get('PR-AUC')


def _overlaps(first, second):
    if not first or not second:
        return False
    try:
        a_low, a_high = float(first[0]), float(first[1])
        b_low, b_high = float(second[0]), float(second[1])
    except (TypeError, ValueError, IndexError):
        return False
    if any(np.isnan(v) for v in (a_low, a_high, b_low, b_high)):
        return False
    return a_low <= b_high and b_low <= a_high


def estimate_cost(n_models, n_repeats, search_budget, search_sampler,
                  n_outer=DEFAULT_OUTER_SPLITS, n_inner=DEFAULT_INNER_SPLITS):
    """Rough fit count, so an hour-long sweep is a decision and not a surprise."""
    budget = max(int(search_budget or 1), 1)
    if search_sampler:
        budget *= SAMPLER_SEARCH_FACTOR
    per_model = int(n_repeats) * int(n_outer) * int(n_inner) * budget
    return per_model * max(int(n_models), 0), per_model


def estimate_reducer_fits(n_repeats, n_outer=DEFAULT_OUTER_SPLITS,
                          n_inner=DEFAULT_INNER_SPLITS):
    """How many times a per-fold reducer is fit, for ONE model.

    Deliberately independent of the search budget: the reducer is part of the
    hyperparameter-INDEPENDENT prefix that build_preprocessor() fits once per
    inner split and every candidate then reuses. One fit per inner split
    (n_outer x n_inner) plus one per outer fold for the final refit, all times
    n_repeats - plus the three threshold-calibration fits each outer fold makes,
    which use the full estimator pipeline and so carry the reducer too.
    """
    n_repeats, n_outer, n_inner = int(n_repeats), int(n_outer), int(n_inner)
    headline = n_repeats * (n_outer * n_inner + n_outer)
    calibration = n_repeats * n_outer * 3
    return headline, calibration


# The engine's own default is a floor, not a ceiling: repeating the outer CV is
# what makes the CIs wide enough to be honest about fold noise.
DEFAULT_REPEATS_UI = max(3, DEFAULT_N_REPEATS)


class ClassComparisonRun:
    """One sweep: the loaded data, the progress counters, and the results so far.

    Mutated by the worker thread and read by the polling callback, so every
    touch of the shared fields goes through the lock.
    """

    def __init__(self, models, settings):
        self._lock = threading.Lock()
        self.models = list(models)
        self.settings = settings
        self.done = 0
        self.total = len(models)
        self.current = None
        self.results = []
        self.finished = False
        self.fatal_error = None
        # Wall-clock is part of the thesis record: a leaderboard without the cost
        # of producing it cannot be reproduced or budgeted for.
        self.started_at = datetime.now()
        self.finished_at = None
        self.saved_paths = {}
        self.save_error = None

    def snapshot(self):
        with self._lock:
            return {'models': list(self.models),
                    'done': self.done, 'total': self.total, 'current': self.current,
                    'results': list(self.results), 'finished': self.finished,
                    'fatal_error': self.fatal_error, 'settings': dict(self.settings),
                    'started_at': self.started_at, 'finished_at': self.finished_at,
                    'saved_paths': dict(self.saved_paths),
                    'save_error': self.save_error}

    def start_model(self, class_type):
        with self._lock:
            self.current = _label_for(class_type)

    def add_result(self, result):
        with self._lock:
            self.results.append(result)
            self.done += 1

    def stop_clock(self):
        with self._lock:
            if self.finished_at is None:
                self.finished_at = datetime.now()

    def record_save(self, paths, error=None):
        with self._lock:
            if paths:
                self.saved_paths = dict(paths)
            self.save_error = error

    def finish(self, fatal_error=None):
        with self._lock:
            self.fatal_error = fatal_error
            self.current = None
            self.finished_at = self.finished_at or datetime.now()
            self.finished = True


def run_sweep(run, on_model=None):
    settings = run.settings
    try:
        # Loaded ONCE. Re-loading per model would reshuffle nothing but would
        # re-derive the frames, and the promise of identical folds depends on
        # every model receiving the very same rows in the very same order.
        (orig_df, orig_feature_col_name, imp_df,
         imp_feature_col_name, features) = load_data_traces(
            settings['input_type'], settings['isImputed'], settings['loader_unbalanced'],
            run.models[0], settings['data_source'], settings['n_components'],
            settings['addGLS'], settings['addFeatEcho'], None, settings['class_input'])
    except Exception:
        run.finish(fatal_error=f'Data loading failed:\n{traceback.format_exc()}')
        return

    for class_type in run.models:
        run.start_model(class_type)
        try:
            result = evaluate_classifier(
                orig_df, orig_feature_col_name, imp_df, imp_feature_col_name,
                settings['unbalanced_type'], class_type, features,
                n_repeats=settings['n_repeats'],
                n_outer_splits=DEFAULT_OUTER_SPLITS,
                n_inner_splits=DEFAULT_INNER_SPLITS,
                search_budget=settings['search_budget'],
                random_state=COMPARISON_RANDOM_STATE,
                scale_features=settings['scale_features'],
                class_input=settings['class_input'],
                beta=settings['beta'],
                search_sampler=settings['search_sampler'],
                reducer=settings.get('reducer'),
                reducer_components=settings.get('reducer_components'),
                # The sweep writes one aggregated report covering every model;
                # per-model snapshots would be redundant noise in reports/.
                autosave=False)
        except Exception:
            # The contract says evaluate_classifier never raises; if it does the
            # sweep still has to survive and say which model broke.
            result = {'class_type': class_type, 'error': traceback.format_exc()}
        run.add_result(result)
        save_run_report(run, partial=True)
        if on_model:
            on_model(run)

    run.stop_clock()
    # Saved before finish(): the polling callback disables itself on the snapshot
    # that first reports 'finished', so the paths must already be in it. A sweep
    # costs hours - it must survive the browser tab being closed.
    save_run_report(run)
    run.finish()


def rank_results(results):
    """Best-first by pooled PR-AUC, with every model whose CI overlaps the
    leader's flagged as tied.

    Without the tie flag a 0.002 PR-AUC gap reads as a winner, which is exactly
    the false ordering the CIs exist to prevent.
    """
    scored = [r for r in results if not r.get('error') and _pr_auc(r) is not None]
    failed = [r for r in results if r.get('error') or _pr_auc(r) is None]
    scored.sort(key=_pr_auc, reverse=True)

    leader_ci = _pr_auc_ci(scored[0]) if scored else None
    ranked = []
    for position, result in enumerate(scored, start=1):
        tied = position > 1 and _overlaps(_pr_auc_ci(result), leader_ci)
        ranked.append({'rank': position, 'result': result, 'tied': tied})
    return ranked, failed


LEADERBOARD_COLUMNS = ['#', 'Model', 'Sampler', 'PR-AUC (95% CI)', 'Lift vs baseline', 'AUC',
                       'Balanced Acc', 'MCC', 'Recall', 'Precision', 'Specificity', 'Brier',
                       'Threshold', 'n_fits']


def _pooled_arrays(result):
    pooled = result.get('pooled') or {}
    y_true = pooled.get('y_true')
    y_proba = pooled.get('y_proba')
    if not y_true or not y_proba or len(y_true) != len(y_proba):
        return None, None
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return None, None
    return y_true, y_proba


# --------------------------------------------------------------------------
# Report persistence
# --------------------------------------------------------------------------

# The keys kept in the archival JSON. 'pooled', 'shap_values', 'shap_input_df'
# and 'shap_df' are deliberately dropped: they are megabytes of per-row arrays,
# and none of them is needed to rebuild the leaderboard.
REPORT_RESULT_KEYS = ('class_type', 'metrics', 'ci', 'pooled_metrics', 'best_params',
                      'sampler', 'threshold', 'n_fits', 'n_folds', 'cm', 'error')

# Column order of the CSV, mirroring the on-screen leaderboard so a row can be
# pasted straight into a thesis table.
REPORT_COLUMNS = ['rank', 'model', 'sampler', 'PR-AUC', 'PR-AUC_CI_low', 'PR-AUC_CI_high',
                  'lift_over_baseline', 'AUC', 'BalancedAccuracy', 'MCC', 'Recall',
                  'Precision', 'Specificity', 'Brier', 'threshold', 'n_fits',
                  'tied_with_best', 'error']


def _jsonable(value):
    """Recursively coerce a result dict into types json.dump accepts.

    The dicts come straight out of sklearn, so they are full of np.float64 /
    np.int64 scalars and ndarray confusion matrices, none of which are JSON
    types. str()-ing them would produce a file that loads but whose numbers are
    no longer numbers, so every scalar is converted through its Python type.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        # NaN/Infinity are not valid JSON; null survives a strict json.load,
        # the bare NaN literal does not.
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_jsonable(row) for row in value.to_dict(orient='records')]
    if isinstance(value, pd.Series):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    # Last resort, and only reachable for things that have no numeric content
    # anyway - best_params can hold estimator objects, whose repr is the record.
    return str(value)


def _ci_bounds(bounds):
    if not bounds:
        return None, None
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    low = None if np.isnan(low) else low
    high = None if np.isnan(high) else high
    return low, high


def _clean_number(value):
    """None for anything that is not a finite number, so CSV cells stay empty."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _first_line(text):
    lines = [line for line in str(text).strip().splitlines() if line.strip()]
    return lines[-1].strip() if lines else ''


def build_report_rows(results):
    """The leaderboard as plain dicts - the shared source for CSV and Markdown."""
    ranked, failed = rank_results(results)
    rows = []

    for entry in ranked:
        result = entry['result']
        metrics = result.get('metrics') or {}
        pr_auc = _pr_auc(result)
        baseline = _baseline(result)
        low, high = _ci_bounds(_pr_auc_ci(result))
        rows.append({
            'rank': entry['rank'],
            'model': _label_for(result.get('class_type')),
            'sampler': result.get('sampler') or '',
            'PR-AUC': pr_auc,
            'PR-AUC_CI_low': low,
            'PR-AUC_CI_high': high,
            'lift_over_baseline': (round(pr_auc / baseline, 4)
                                   if (baseline and pr_auc is not None) else None),
            'AUC': _clean_number(metrics.get('AUC')),
            'BalancedAccuracy': _clean_number(metrics.get('BalancedAccuracy')),
            'MCC': _clean_number(metrics.get('MCC')),
            'Recall': _clean_number(metrics.get('Recall')),
            'Precision': _clean_number(metrics.get('Precision')),
            'Specificity': _clean_number(metrics.get('Specificity')),
            'Brier': _clean_number(metrics.get('Brier')),
            'threshold': _clean_number(result.get('threshold')),
            'n_fits': int(result.get('n_fits') or 0),
            'tied_with_best': bool(entry['tied']),
            'error': '',
        })

    # A model that blew up is a result too: it has to appear in the saved record
    # or the report silently claims fewer models were compared than were asked for.
    for result in failed:
        row = {column: None for column in REPORT_COLUMNS}
        row.update({
            'rank': None,
            'model': _label_for(result.get('class_type')),
            'sampler': result.get('sampler') or '',
            'n_fits': int(result.get('n_fits') or 0),
            'tied_with_best': False,
            'error': _first_line(result.get('error') or 'produced no PR-AUC'),
        })
        rows.append(row)

    return rows


def build_report_frame(results):
    frame = pd.DataFrame(build_report_rows(results), columns=REPORT_COLUMNS)
    # Nullable ints: a failed model has no rank, and plain NaN would turn the
    # whole column into floats and print the winner as "1.0" in the thesis table.
    for column in ('rank', 'n_fits'):
        frame[column] = frame[column].astype('Int64')
    return frame


def build_run_config(state, partial=False):
    """Everything needed to reproduce the sweep. A score without its setup is
    not evidence, so this block goes into all three formats."""
    settings = state.get('settings') or {}
    results = state.get('results') or []
    started = state.get('started_at')
    finished = state.get('finished_at') or datetime.now()
    duration = (finished - started).total_seconds() if started else None

    baseline = next((_baseline(r) for r in results if _baseline(r) is not None), None)
    prevalence = next(((r.get('pooled_metrics') or {}).get('Prevalence')
                       for r in results if (r.get('pooled_metrics') or {}).get('Prevalence')
                       is not None), baseline)

    search_sampler = bool(settings.get('search_sampler'))
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'started_at': started.isoformat(timespec='seconds') if started else None,
        'finished_at': finished.isoformat(timespec='seconds'),
        'duration_seconds': round(duration, 1) if duration is not None else None,
        'duration_hms': (f'{int(duration) // 3600:d}h '
                         f'{int(duration) % 3600 // 60:02d}m '
                         f'{int(duration) % 60:02d}s') if duration is not None else None,
        'partial': bool(partial),
        'models_requested': [_label_for(m) for m in (state.get('models') or [])],
        'models_completed': len(results),
        'data_source': settings.get('data_source'),
        'classify_by': settings.get('class_input'),
        'concat_type': settings.get('input_type'),
        'n_components': settings.get('n_components'),
        'imputed': bool(settings.get('isImputed')),
        'add_gls': bool(settings.get('addGLS')),
        'add_echo_features': bool(settings.get('addFeatEcho')),
        'standardisation_mode': settings.get('scaling_mode'),
        'scale_features': bool(settings.get('scale_features')),
        # Part of the model, not of the data prep: refit inside every split.
        'reducer': settings.get('reducer') or 'none',
        'reducer_components': (settings.get('reducer_components')
                               if settings.get('reducer') else None),
        'imbalance_strategy': 'search' if search_sampler else settings.get('unbalanced_type'),
        'sampler_searched_per_fold': search_sampler,
        'n_repeats': settings.get('n_repeats'),
        'n_outer_splits': DEFAULT_OUTER_SPLITS,
        'n_inner_splits': DEFAULT_INNER_SPLITS,
        'search_budget': settings.get('search_budget'),
        'beta': settings.get('beta'),
        'random_state': COMPARISON_RANDOM_STATE,
        'total_n_fits': sum(int(r.get('n_fits') or 0) for r in results),
        'prevalence': _clean_number(prevalence),
        'pr_auc_baseline': _clean_number(baseline),
    }


def build_verdict(rows, config):
    """The one line the reader quotes - and it must not invent a winner.

    When the runner-up's CI overlaps the leader's, the ordering between them is
    fold noise; saying "best: X" without naming the tie would be a claim the
    numbers do not support.
    """
    scored = [row for row in rows if row.get('rank')]
    if not scored:
        return 'No model produced a PR-AUC - nothing to rank.'

    best = scored[0]
    tied = [row['model'] for row in scored if row.get('tied_with_best')]
    ci = ''
    if best['PR-AUC_CI_low'] is not None and best['PR-AUC_CI_high'] is not None:
        ci = (f" (95% CI {best['PR-AUC_CI_low']:.2f}-{best['PR-AUC_CI_high']:.2f})")
    baseline = config.get('pr_auc_baseline')
    against = f' vs {baseline:.2f} prevalence baseline' if baseline else ''

    verdict = (f"Best: {best['model']} - PR-AUC "
               f"{_fmt(best['PR-AUC'], 2)}{ci}{against}")
    if tied:
        verdict += f", statistically tied with {', '.join(tied)}"
    return verdict


def _md_cell(value, digits=3):
    if value is None or value == '':
        return '-'
    if isinstance(value, bool):
        return 'yes' if value else ''
    if isinstance(value, (int, np.integer)):
        return f'{int(value):,}'
    if isinstance(value, float):
        return _fmt(value, digits)
    return str(value)


def build_markdown_report(rows, config, partial=False):
    scored = [row for row in rows if row.get('rank')]
    lines = ['# Classifier comparison report', '']
    lines.append(f'**{build_verdict(rows, config)}**')
    lines.append('')
    if partial:
        lines.append('> PARTIAL run - saved while the sweep was still going, '
                     f"{config['models_completed']} of "
                     f"{len(config['models_requested'])} models finished.")
        lines.append('')

    if scored:
        best = scored[0]
        lines.append(f"## Winner: {best['model']}")
        lines.append('')
        lines.append(f"- Pooled PR-AUC: **{_md_cell(best['PR-AUC'])}** "
                     f"(95% CI {_md_cell(best['PR-AUC_CI_low'])}-"
                     f"{_md_cell(best['PR-AUC_CI_high'])})")
        lines.append(f"- PR-AUC baseline (prevalence): "
                     f"{_md_cell(config.get('pr_auc_baseline'))} - "
                     f"lift {_md_cell(best['lift_over_baseline'], 2)}x")
        lines.append(f"- AUC {_md_cell(best['AUC'])} | Balanced accuracy "
                     f"{_md_cell(best['BalancedAccuracy'])} | MCC {_md_cell(best['MCC'])}")
        lines.append(f"- Recall {_md_cell(best['Recall'])} | Precision "
                     f"{_md_cell(best['Precision'])} | Specificity "
                     f"{_md_cell(best['Specificity'])} | Brier {_md_cell(best['Brier'])}")
        lines.append(f"- Sampler: {_md_cell(best['sampler'])} | operating threshold "
                     f"{_md_cell(best['threshold'])}")
        tied = [row['model'] for row in scored if row.get('tied_with_best')]
        if tied:
            lines.append(f"- **Statistically tied with {', '.join(tied)}** - their "
                         'PR-AUC CIs overlap the leader\'s, so the ordering between '
                         'them is noise.')
        lines.append('')

    lines.append('## Leaderboard')
    lines.append('')
    header = ['#', 'Model', 'Tied', 'Sampler', 'PR-AUC', '95% CI', 'Lift', 'AUC',
              'Bal. Acc', 'MCC', 'Recall', 'Precision', 'Specificity', 'Brier',
              'Threshold', 'n_fits', 'Error']
    lines.append('| ' + ' | '.join(header) + ' |')
    lines.append('|' + '---|' * len(header))
    for row in rows:
        ci = '-'
        if row['PR-AUC_CI_low'] is not None or row['PR-AUC_CI_high'] is not None:
            ci = f"[{_md_cell(row['PR-AUC_CI_low'])}, {_md_cell(row['PR-AUC_CI_high'])}]"
        lift = ('-' if row['lift_over_baseline'] is None
                else f"{row['lift_over_baseline']:.2f}x")
        lines.append('| ' + ' | '.join([
            _md_cell(row['rank']), row['model'], _md_cell(row['tied_with_best']),
            _md_cell(row['sampler']), _md_cell(row['PR-AUC']), ci, lift,
            _md_cell(row['AUC']), _md_cell(row['BalancedAccuracy']), _md_cell(row['MCC']),
            _md_cell(row['Recall']), _md_cell(row['Precision']),
            _md_cell(row['Specificity']), _md_cell(row['Brier']),
            _md_cell(row['threshold']), _md_cell(row['n_fits']),
            _md_cell(row['error']),
        ]) + ' |')
    lines.append('')

    lines.append('## Run configuration')
    lines.append('')
    lines.append('| Setting | Value |')
    lines.append('|---|---|')
    for key, value in config.items():
        if isinstance(value, list):
            value = ', '.join(str(v) for v in value)
        lines.append(f'| {key} | {"-" if value is None or value == "" else value} |')
    lines.append('')
    return '\n'.join(lines)


def build_json_report(state, config, rows):
    """The archival artefact: enough to rebuild the leaderboard without re-running."""
    return {
        'report_type': 'classifier_comparison',
        'schema_version': 1,
        'config': _jsonable(config),
        'verdict': build_verdict(rows, config),
        'leaderboard': _jsonable(rows),
        'results': [_jsonable({key: result.get(key) for key in REPORT_RESULT_KEYS})
                    for result in (state.get('results') or [])],
    }


def _safe_token(value, fallback='na'):
    token = ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in str(value or ''))
    return token.strip('-') or fallback


def report_basename(state, partial=False):
    settings = state.get('settings') or {}
    stamp = (state.get('finished_at') or datetime.now()).strftime('%Y%m%d-%H%M%S')
    suffix = '_partial' if partial else ''
    return (f"classifier_comparison_{stamp}_"
            f"{_safe_token(settings.get('class_input'))}_"
            f"{_safe_token(settings.get('data_source'))}{suffix}")


def write_comparison_report(state, partial=False, reports_dir=None):
    """Write the JSON, CSV and Markdown report. Returns (paths, error_message).

    Each format is written under its own guard: a failure on one (a locked CSV
    open in Excel is the usual one) must not cost the other two, and no failure
    may ever propagate into the sweep thread or the callback.
    """
    directory = Path(reports_dir) if reports_dir is not None else REPORTS_DIR
    results = state.get('results') or []
    paths, errors = {}, []

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return {}, f'Could not create {directory}: {exc}'

    config = build_run_config(state, partial=partial)
    rows = build_report_rows(results)
    base = directory / report_basename(state, partial=partial)

    try:
        payload = build_json_report(state, config, rows)
        with open(f'{base}.json', 'w', encoding='utf-8') as handle:
            # allow_nan=False: a NaN would make the file unreadable by anything
            # stricter than Python's own json, and _jsonable already nulls them.
            json.dump(payload, handle, indent=2, allow_nan=False)
        paths['json'] = f'{base}.json'
    except Exception as exc:
        errors.append(f'JSON: {exc}')

    try:
        build_report_frame(results).to_csv(f'{base}.csv', index=False)
        paths['csv'] = f'{base}.csv'
    except Exception as exc:
        errors.append(f'CSV: {exc}')

    try:
        with open(f'{base}.md', 'w', encoding='utf-8') as handle:
            handle.write(build_markdown_report(rows, config, partial=partial))
        paths['markdown'] = f'{base}.md'
    except Exception as exc:
        errors.append(f'Markdown: {exc}')

    return paths, ('; '.join(errors) if errors else None)


def save_run_report(run, partial=False):
    """Persist a run and record the outcome on it. Never raises: losing the save
    must never also lose the in-memory results."""
    try:
        state = run.snapshot()
        paths, error = write_comparison_report(state, partial=partial)
    except Exception:
        run.record_save({}, error=_first_line(traceback.format_exc()))
        return {}, run.snapshot()['save_error']
    run.record_save(paths, error=error)
    return paths, error


