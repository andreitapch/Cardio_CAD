"""Static checks that the viewer's callbacks are wired to the session store.

Dash resolves callback ids at runtime, so a State that was never added - or a
global that crept back in - only shows up as a mixed-up figure after a
multi-minute run. Parsing the module catches it immediately.
"""
import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
VIEWER = 'dash_viewer/dimensionality_reduction/dimen_reduction_viewer.py'
SESSION_STORE_ID = 'dm-session-id'


@pytest.fixture(scope='module')
def tree():
    return ast.parse((REPO_ROOT / VIEWER).read_bytes().decode('utf-8'))


@pytest.fixture(scope='module')
def source():
    return (REPO_ROOT / VIEWER).read_bytes().decode('utf-8')


def callbacks(tree):
    """Every @callback-decorated function, with the source of its decorator."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                name = getattr(decorator.func, 'id', None) if isinstance(decorator, ast.Call) else None
                if name == 'callback':
                    yield node, decorator


def test_no_module_level_globals_remain(tree):
    """Run state in a global is shared by every tab and every visitor."""
    declared = [name for node in ast.walk(tree) if isinstance(node, ast.Global)
                for name in node.names]

    assert declared == [], f'global state is back: {declared}'


def test_layout_declares_the_session_store(tree):
    store_ids = {kw.value.value
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 for kw in node.keywords
                 if kw.arg == 'id' and isinstance(kw.value, ast.Constant)}

    assert SESSION_STORE_ID in store_ids, 'the page must carry a session id'


def test_every_callback_reading_run_state_takes_a_session_id(tree, source):
    """A callback that touches a run must say which session's run it means."""
    offenders = []
    for node, decorator in callbacks(tree):
        body = ast.get_source_segment(source, node) or ''
        reads_run_state = '_runs.get(' in body or '_runs.put(' in body
        if not reads_run_state:
            continue
        declares_session = SESSION_STORE_ID in (ast.get_source_segment(source, decorator) or '')
        takes_argument = 'session_id' in [arg.arg for arg in node.args.args]
        if not (declares_session and takes_argument):
            offenders.append(node.name)

    assert offenders == [], f'callbacks missing the session id: {offenders}'


def test_run_is_published_only_after_it_completes(tree, source):
    """A half-built run must not be visible to the plotting callbacks."""
    for node, _ in callbacks(tree):
        body = ast.get_source_segment(source, node) or ''
        if '_runs.put(' not in body:
            continue
        put_at = body.index('_runs.put(')
        assert 'trigger_callbacks(' in body[:put_at], (
            f'{node.name} publishes the run before computing it')


def test_missing_run_is_handled_rather_than_raising(tree, source):
    """Plot and save callbacks fire before any run exists."""
    for node, _ in callbacks(tree):
        body = ast.get_source_segment(source, node) or ''
        if '_runs.get(' not in body:
            continue
        assert 'is not None' in body or 'is None' in body, (
            f'{node.name} does not check whether a run exists')
