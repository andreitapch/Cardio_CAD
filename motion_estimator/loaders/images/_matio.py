"""Uniform access to MATLAB ``.mat`` files, whether v5 or v7.3.

The dataset mixes both: the small files (``scar_struct.mat``, the GT_STRAIN
files, and the ``.srd`` clips themselves) are classic v5 MAT-files that
``scipy.io.loadmat`` reads directly, while ``pack_SRD_HDPULSE.m`` switches to
``-v7.3`` -- i.e. HDF5 -- whenever ``USData`` exceeds 1 GB.

Two shapes therefore come back from the two backends: scipy hands over nested
structured/object ``ndarray``s, h5py hands over nested groups. :func:`field`
and friends smooth that over so callers never branch on the backend.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

__all__ = ["load_mat", "field", "fields", "has_field", "text", "scalar", "unwrap"]

_V73_MAGIC = b"\x89HDF\r\n\x1a\n"


def is_hdf5(path: str | Path) -> bool:
    """True when *path* is an HDF5 container (a ``-v7.3`` MAT-file)."""
    with open(path, "rb") as handle:
        return handle.read(8) == _V73_MAGIC


def load_mat(path: str | Path) -> dict:
    """Load *path* and return its top-level variables, MATLAB axis order kept.

    v7.3 files store arrays transposed relative to MATLAB, so this reverses
    their axes on the way out. That means ``load_mat`` output is directly
    comparable between the two formats -- which is what lets the ports in this
    package index exactly as the original ``.m`` files do.
    """
    path = Path(path)
    if is_hdf5(path):
        return _load_v73(path)
    raw = sio.loadmat(path, struct_as_record=True, squeeze_me=False)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _load_v73(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as handle:
        return {
            key: _from_h5(handle[key], handle)
            for key in handle.keys()
            if key != "#refs#"
        }


def _from_h5(node, root):
    import h5py

    if isinstance(node, h5py.Group):
        return {key: _from_h5(node[key], root) for key in node.keys()}

    if h5py.check_dtype(ref=node.dtype) is not None or node.dtype.kind == "O":
        resolved = [_from_h5(root[ref], root) for ref in np.asarray(node).ravel()]
        if len(resolved) == 1:
            return resolved[0]
        return np.asarray(resolved, dtype=object).reshape(np.asarray(node).shape[::-1])

    values = np.asarray(node)
    if node.attrs.get("MATLAB_class", b"") == b"char":
        return "".join(chr(int(c)) for c in values.ravel() if int(c) != 0)
    # MATLAB writes v7.3 arrays with reversed dimension order.
    return values.T if values.ndim > 1 else values


def unwrap(value):
    """Strip the size-1 object wrappers scipy puts around every struct field."""
    array = np.asarray(value)
    while array.dtype == object and array.dtype.names is None and array.size == 1:
        array = np.asarray(array.item())
    return array


def fields(struct) -> tuple[str, ...]:
    """Field names of *struct*, for either backend."""
    if isinstance(struct, dict):
        return tuple(struct.keys())
    array = unwrap(struct)
    return tuple(array.dtype.names or ())


def has_field(struct, name: str) -> bool:
    return name in fields(struct)


def field(struct, name: str):
    """Value of field *name* on *struct*, unwrapped once."""
    if isinstance(struct, dict):
        if name not in struct:
            raise KeyError(f"no field {name!r}; present: {sorted(struct)}")
        return struct[name]

    array = unwrap(struct)
    if array.dtype.names is None:
        raise TypeError(f"not a MATLAB struct (dtype {array.dtype}), cannot read {name!r}")
    if name not in array.dtype.names:
        raise KeyError(f"no field {name!r}; present: {list(array.dtype.names)}")
    return unwrap(array.ravel()[0][name])


def text(value) -> str:
    """Coerce a MATLAB char field to ``str``, empty char arrays included."""
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.size == 0:
        return ""
    if array.dtype.kind in "US":
        return str(array.ravel()[0])
    return "".join(chr(int(c)) for c in array.ravel() if int(c) != 0)


def scalar(value):
    """Coerce a 1x1 MATLAB numeric field to a plain Python number."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected a scalar, got shape {array.shape}")
    return array.ravel()[0].item()
