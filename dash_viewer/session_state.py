"""Per-browser-session storage for viewer run state.

Dash callbacks are plain module-level functions, so viewers have kept the object
produced by a run in a module global. That makes the run state shared by every
browser tab and every visitor hitting the same server: a second run overwrites
the first while its plotting callbacks are still reading, and the resulting
figure silently mixes two runs.

A run holds fitted models and full trace frames, so it cannot go through
dcc.Store (which serialises to the browser). Instead the page holds only a
session id and the objects stay here, keyed by that id.
"""
import threading
import uuid
from collections import OrderedDict

DEFAULT_MAX_RUNS = 4

# Used when a page predates the session store and sends no id. Behaves like the
# old global - shared - but keeps the callbacks from raising.
FALLBACK_KEY = '__no_session__'


def new_session_id():
    """Fresh id for a page load. Each open tab gets its own run state."""
    return str(uuid.uuid4())


class RunStore:
    """Bounded, thread-safe map of session id -> run state.

    Bounded because each entry can hold a fitted model and every trace in the
    cohort; without a cap a long-lived server accumulates them until it runs out
    of memory. The least recently used entry is dropped first.
    """

    def __init__(self, max_runs=DEFAULT_MAX_RUNS):
        if max_runs < 1:
            raise ValueError(f'max_runs must be at least 1, got {max_runs}')
        self._max_runs = max_runs
        self._runs = OrderedDict()
        self._lock = threading.RLock()

    def put(self, session_id, run):
        """Store run for this session, evicting the least recently used if full."""
        key = session_id or FALLBACK_KEY
        with self._lock:
            self._runs[key] = run
            self._runs.move_to_end(key)
            while len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)
        return run

    def get(self, session_id):
        """Return this session's run, or None if it has not run one yet."""
        key = session_id or FALLBACK_KEY
        with self._lock:
            run = self._runs.get(key)
            if run is not None:
                self._runs.move_to_end(key)
            return run

    def clear(self):
        with self._lock:
            self._runs.clear()

    def __len__(self):
        with self._lock:
            return len(self._runs)
