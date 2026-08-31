"""Static checks on the classification viewer's Dash wiring.

Dash resolves callback ids at runtime, so a typo in an Output id or a return
tuple that is one element short only shows up when the page is opened and the
button pressed - after a multi-minute classification run. These parse the
module instead, so the mistakes surface immediately.

No dash install required.
"""
import ast
import pathlib

import pytest

VIEWERS = [
    'dash_viewer/classification/machine_learning/best_params_classifiers/best_knn_class_viewer.py',
]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def load(path):
    return ast.parse((REPO_ROOT / path).read_bytes().decode('utf-8'))


def declared_component_ids(tree):
    """Every id='...' keyword passed to a component constructor in the layout."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == 'id' and isinstance(kw.value, ast.Constant):
                    found.add(kw.value.value)
    return found


def callback_specs(tree):
    """(function, outputs, inputs_and_states) for each @callback-decorated function."""
    specs = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not (isinstance(dec, ast.Call) and getattr(dec.func, 'id', None) == 'callback'):
                continue
            outputs, inputs = [], []
            for arg in dec.args:
                if not isinstance(arg, ast.Call):
                    continue
                kind = getattr(arg.func, 'id', None)
                if not arg.args or not isinstance(arg.args[0], ast.Constant):
                    continue
                target = outputs if kind == 'Output' else inputs
                target.append(arg.args[0].value)
            specs.append((node, outputs, inputs))
    return specs


def return_widths(func):
    """Width of every return statement in `func`, ignoring nested functions."""
    widths = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Tuple):
                widths.append((node.lineno, len(value.elts)))
            elif isinstance(value, ast.BinOp) and isinstance(value.op, ast.Mult):
                # the (dash.no_update,) * N idiom
                right = value.right
                if isinstance(right, ast.Constant) and isinstance(right.value, int):
                    widths.append((node.lineno, right.value))
            elif isinstance(value, ast.Constant) and value.value is None:
                widths.append((node.lineno, 0))
    return widths


@pytest.mark.parametrize('path', VIEWERS)
def test_every_callback_id_exists_in_the_layout(path):
    tree = load(path)
    declared = declared_component_ids(tree)

    missing = []
    for func, outputs, inputs in callback_specs(tree):
        for cid in outputs + inputs:
            if cid not in declared:
                missing.append(f'{func.name}: {cid}')

    assert not missing, 'callback ids with no matching component: ' + ', '.join(missing)


@pytest.mark.parametrize('path', VIEWERS)
def test_return_arity_matches_output_count(path):
    tree = load(path)

    problems = []
    for func, outputs, _ in callback_specs(tree):
        if len(outputs) <= 1:
            continue  # a single Output may be returned bare
        for lineno, width in return_widths(func):
            if width != len(outputs):
                problems.append(
                    f'{func.name} line {lineno}: returns {width} value(s), '
                    f'{len(outputs)} Outputs declared')

    assert not problems, '; '.join(problems)


@pytest.mark.parametrize('path', VIEWERS)
def test_dropdown_defaults_are_selectable(path):
    """A dropdown whose value is not among its options renders blank."""
    tree = load(path)

    problems = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'Dropdown'):
            continue

        kwargs = {kw.arg: kw.value for kw in node.keywords}
        options, value = kwargs.get('options'), kwargs.get('value')
        if not isinstance(options, ast.List) or not options.elts:
            continue
        if not isinstance(value, ast.Constant) or value.value is None:
            continue

        allowed = set()
        for element in options.elts:
            if isinstance(element, ast.Dict):
                for k, v in zip(element.keys, element.values):
                    if isinstance(k, ast.Constant) and k.value == 'value' \
                            and isinstance(v, ast.Constant):
                        allowed.add(v.value)
        if allowed and value.value not in allowed:
            cid = kwargs.get('id')
            name = cid.value if isinstance(cid, ast.Constant) else f'line {node.lineno}'
            problems.append(f"{name}: default {value.value!r} not in {sorted(allowed)}")

    assert not problems, '; '.join(problems)
