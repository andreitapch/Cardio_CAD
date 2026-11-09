"""The per-session run store that replaced the module-level global."""
import threading

import pytest

from dash_viewer.session_state import FALLBACK_KEY, RunStore, new_session_id


def test_a_session_gets_its_own_run_back():
    store = RunStore()
    session = new_session_id()

    store.put(session, 'run-a')

    assert store.get(session) == 'run-a'


def test_two_sessions_do_not_see_each_other():
    """The whole point: two open tabs must not share a fitted model."""
    store = RunStore()
    first, second = new_session_id(), new_session_id()

    store.put(first, 'run-a')
    store.put(second, 'run-b')

    assert store.get(first) == 'run-a'
    assert store.get(second) == 'run-b'


def test_session_ids_are_unique():
    assert len({new_session_id() for _ in range(100)}) == 100


def test_unknown_session_returns_none_rather_than_raising():
    """Plot callbacks fire before any run; they must degrade, not blow up."""
    store = RunStore()

    assert store.get(new_session_id()) is None


def test_missing_session_id_falls_back_to_a_shared_slot():
    """A page with no store still works - it just shares, as the global did."""
    store = RunStore()

    store.put(None, 'run-a')

    assert store.get(None) == 'run-a'
    assert store.get(FALLBACK_KEY) == 'run-a'


def test_least_recently_used_run_is_evicted():
    """Each run holds a model and every trace in the cohort; the map is bounded."""
    store = RunStore(max_runs=2)

    store.put('one', 'run-1')
    store.put('two', 'run-2')
    store.put('three', 'run-3')

    assert store.get('one') is None
    assert store.get('two') == 'run-2'
    assert store.get('three') == 'run-3'
    assert len(store) == 2


def test_reading_a_run_keeps_it_alive():
    store = RunStore(max_runs=2)
    store.put('one', 'run-1')
    store.put('two', 'run-2')

    store.get('one')          # refreshes recency
    store.put('three', 'run-3')

    assert store.get('one') == 'run-1', 'the run just read should not be the one dropped'
    assert store.get('two') is None


def test_replacing_a_session_run_does_not_grow_the_store():
    store = RunStore(max_runs=2)

    store.put('one', 'run-1')
    store.put('one', 'run-2')

    assert len(store) == 1
    assert store.get('one') == 'run-2'


def test_max_runs_must_be_positive():
    with pytest.raises(ValueError):
        RunStore(max_runs=0)


def test_concurrent_access_is_safe():
    """Dash serves callbacks from a thread pool; puts and gets overlap."""
    store = RunStore(max_runs=8)
    errors = []

    def hammer(n):
        try:
            for i in range(200):
                store.put(f'session-{n}', f'run-{n}-{i}')
                store.get(f'session-{n}')
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, f'concurrent access raised: {errors[:3]}'
    assert len(store) <= 8
