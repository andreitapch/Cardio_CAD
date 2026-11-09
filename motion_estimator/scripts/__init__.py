"""Command-line entry points for the motion estimator.

Importing this package sets ``MKL_THREADING_LAYER`` and pulls torch in before
anything MKL-linked, because ``python -m motion_estimator.scripts.<name>`` loads
neither ``conftest.py`` nor ``dash_main.py``, which carry the same two lines.

Without it, a script that touches scipy or numpy before torch aborts with
``OMP: Error #15`` -- or, if torch loses the race entirely, fails to load its
DLLs with ``WinError 127``. The trigger is a BLAS call rather than an import;
see ``docs/environment-notes.md``.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # torch genuinely absent: several scripts here do not need it, so this is
    # not fatal. Anything else -- a half-upgraded install, a DLL that will not
    # load -- must propagate. Swallowing those defeats the whole purpose of
    # this file: the guard stops running, the script imports torch later
    # anyway, and the failure resurfaces as an unreadable C++ abort
    # ("Unhandled exception caught in c10/util/AbortHandler.h") with no Python
    # traceback at all.
    pass
