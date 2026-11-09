"""Every classifier must build a grid for every imbalance strategy.

find_best_class_params wraps the whole search in try/except and returns a row of
Nones, so a grid that fails to build renders as an empty tab with the reason
only in the terminal - which reads as "the model is bad", not "it never ran".
"""
import pytest

from cardio_app.services.classification.model_params import ModelParams

CLASSIFIERS = ['KNN_class', 'DT_class', 'RF_class', 'GBM_class',
               'XGBoost_class', 'LogisticRegression_class']
STRATEGIES = ['None', 'SMOTE', 'ADASYN', 'Balanced']


@pytest.mark.parametrize('class_type', CLASSIFIERS)
@pytest.mark.parametrize('unbalanced_type', STRATEGIES)
def test_grid_builds_for_every_combination(class_type, unbalanced_type):
    params = ModelParams()

    params.get_param_grid(class_type, unbalanced_type, n_iter=3)

    assert params.clf is not None
    assert params.param_combinations, f'{class_type} + {unbalanced_type} produced no candidates'


@pytest.mark.parametrize('class_type', ['RF_class', 'LogisticRegression_class'])
@pytest.mark.parametrize('unbalanced_type', ['SMOTE', 'ADASYN', 'Balanced'])
def test_class_weight_is_dropped_when_a_sampler_balances(class_type, unbalanced_type):
    """Correcting the imbalance twice over-predicts the minority class."""
    params = ModelParams()
    params.get_param_grid(class_type, unbalanced_type, n_iter=None)

    for combination in params.param_combinations:
        weight = combination.get('class_weight', combination.get('clf__class_weight'))
        assert weight is None, f'{class_type} + {unbalanced_type} still passes class_weight={weight}'


@pytest.mark.parametrize('class_type', ['RF_class', 'LogisticRegression_class'])
def test_class_weight_is_used_when_nothing_else_balances(class_type):
    params = ModelParams()
    params.get_param_grid(class_type, 'None', n_iter=None)

    weights = {combination.get('class_weight', combination.get('clf__class_weight'))
               for combination in params.param_combinations}
    assert weights != {None}, f'{class_type} has no imbalance correction at all'


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match='Unknown classifier'):
        ModelParams().get_param_grid('NotAClassifier', 'None')


def test_out_of_bag_scoring_is_not_requested():
    """Nothing reads oob_score_, so computing it was pure cost per fit."""
    params = ModelParams()
    params.get_param_grid('RF_class', 'None', n_iter=None)

    assert all('oob_score' not in combination for combination in params.param_combinations)
