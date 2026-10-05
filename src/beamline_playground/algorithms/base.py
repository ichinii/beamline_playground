"""Common interface shared by all propagation kernels."""

from typing import Protocol

import jax

# One discretised object, as produced by `SceneInstance`: the keys are
# "pos_x", "pos_y", "normal_x", "normal_y" and "dx", each a 1-D array of
# length n (the object's element count).
ObjectSamples = dict[str, jax.Array]

# Memory budget for a single propagation step, counted in source x destination
# matrix entries. At complex64 (8 bytes) the default is roughly 32 MB, which
# keeps a propagation step well inside cache/VRAM on any realistic machine.
# Kernels chunk their destination axis to stay under this.
DEFAULT_MAX_MATRIX_ENTRIES = 1 << 22


class Propagator(Protocol):
    """Propagates a complex field from one discretised object to another.

    Implementations take these four arguments positionally, so that any
    propagator can be substituted for any other. They may add further
    keyword arguments provided every one of them has a default.

    Args:
        k: angular wavenumber, 2*pi/wavelength, in 1/mm.
        src_field: complex field at each source element.
        src_obj: the radiating object's samples.
        dst_obj: the receiving object's samples.

    Returns:
        The complex field contributed to each element of `dst_obj`.
    """

    def __call__(
        self,
        k: jax.Array | float,
        src_field: jax.Array,
        src_obj: ObjectSamples,
        dst_obj: ObjectSamples,
    ) -> jax.Array: ...


def chunk_size_for(n_src: int, n_dst: int, max_entries: int = DEFAULT_MAX_MATRIX_ENTRIES) -> int:
    """How many destination elements to process at once.

    Bounds peak memory at roughly `max_entries` matrix entries regardless of
    how finely the source is sampled.
    """
    per_chunk = max(1, max_entries // max(1, n_src))
    return int(min(n_dst, per_chunk))
