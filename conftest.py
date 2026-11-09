"""Repo-wide pytest setup.

pytest loads this before collecting any test module, which is the only place
early enough to fix the import order: torch and conda's MKL both ship a
libiomp5md.dll, Windows binds that name once per process, and whichever loads
first wins. Collect a test that imports shap before one that imports torch and
torch's shm.dll fails with WinError 127 - so torch goes first, exactly as
dash_viewer/dash_main.py does for the server.
"""
import os

# Same reason as dash_main.py: MKL must not pull conda's libomp.dll in next
# to torch, or the test process aborts instead of failing.
os.environ.setdefault('MKL_THREADING_LAYER', 'TBB')

try:
    import torch  # noqa: F401
except Exception:  # torch is optional for most of the suite
    pass


def pytest_configure(config):
    """Register the markers used across the suite.

    ``slow`` marks tests that download or run a neural checkpoint - the LoFTR
    matcher tests. Deselect them with ``-m "not slow"``.
    """
    config.addinivalue_line(
        "markers", "slow: needs a model checkpoint; deselect with -m 'not slow'"
    )
