"""Interchangeable propagation kernels.

Every propagator here conforms to `Propagator`: it takes
`(k, src_field, src_obj, dst_obj)` and returns the field contributed to the
destination object.
"""

from .base import DEFAULT_MAX_MATRIX_ENTRIES, ObjectSamples, Propagator, chunk_size_for
from .hankel import propagate_mirror as hankel_mirror
from .hankel import propagate_slit as hankel_slit
from .rayleigh_sommerfeld import propagate as rayleigh_sommerfeld

__all__ = [
    "DEFAULT_MAX_MATRIX_ENTRIES",
    "ObjectSamples",
    "Propagator",
    "chunk_size_for",
    "hankel_mirror",
    "hankel_slit",
    "rayleigh_sommerfeld",
]
