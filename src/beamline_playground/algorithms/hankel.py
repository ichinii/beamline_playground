"""Near-field boundary-element kernels built on Hankel functions.

The 2D free-space Green's function is (i/4) H0(kr), so a boundary-element
formulation needs Hankel evaluations. jax has no native Hankel function, so
these are computed on the host via `scipy.special` behind a
`jax.pure_callback`, parallelised across CPU cores.

Unlike `rayleigh_sommerfeld`, these kernels are valid in the near field.
They are still experimental: only the slit and mirror kernels are wired up,
and the hypersingular kernel below is unvalidated.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

from .base import DEFAULT_MAX_MATRIX_ENTRIES, ObjectSamples, chunk_size_for

logger = logging.getLogger(__name__)

_NUM_CPUS = os.cpu_count() or 4
_POOL = ThreadPoolExecutor(max_workers=_NUM_CPUS)

# Below this many elements the thread dispatch costs more than it saves.
_PARALLEL_THRESHOLD = 2048


def _cpu_parallel(fn, x):
    """Evaluate `fn` over `x` across CPU threads, writing into one output buffer."""
    out_dtype = np.complex64 if x.dtype == np.float32 else np.complex128
    out = np.empty(x.shape, dtype=out_dtype)

    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)
    n = len(x_flat)

    if n < _PARALLEL_THRESHOLD:
        fn(x, out=out)
        return out

    chunk_size = (n + _NUM_CPUS - 1) // _NUM_CPUS
    futures = []
    for i in range(_NUM_CPUS):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        if start < end:
            futures.append(_POOL.submit(fn, x_flat[start:end], out=out_flat[start:end]))

    for future in futures:
        future.result()

    return out


def _hankel1(order, kr):
    """Hankel function of the first kind, evaluated on the host."""
    result_shape = jax.ShapeDtypeStruct(
        kr.shape,
        jnp.complex64 if kr.dtype == jnp.float32 else jnp.complex128,
    )
    return jax.pure_callback(
        partial(_cpu_parallel, partial(scipy.special.hankel1, order)),
        result_shape,
        kr,
    )


def _pair_geometry(src_pos_x, src_pos_y, dst_pos_x, dst_pos_y):
    """Separation and unit direction for every (destination, source) pair."""
    vel_x = dst_pos_x[:, jnp.newaxis] - src_pos_x[jnp.newaxis, :]
    vel_y = dst_pos_y[:, jnp.newaxis] - src_pos_y[jnp.newaxis, :]
    # The epsilon keeps the gradient finite when an element sees itself.
    length = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    return length, vel_x / length, vel_y / length


@jax.jit
def _mirror_kernel(
    k,
    src_field,
    src_pos_x,
    src_pos_y,
    src_normal_x,
    src_normal_y,
    src_dx,
    dst_pos_x,
    dst_pos_y,
):
    length, dir_x, dir_y = _pair_geometry(src_pos_x, src_pos_y, dst_pos_x, dst_pos_y)
    cos_theta_src = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])

    h0 = _hankel1(0, k * length)
    # src_dx belongs inside the sum: it is the width of each source element.
    integrand = h0 * src_field * cos_theta_src * src_dx
    return -0.5 * k * jnp.sum(integrand, axis=1)


@jax.jit
def _slit_kernel(
    k,
    src_field,
    src_pos_x,
    src_pos_y,
    src_normal_x,
    src_normal_y,
    src_dx,
    dst_pos_x,
    dst_pos_y,
):
    length, dir_x, dir_y = _pair_geometry(src_pos_x, src_pos_y, dst_pos_x, dst_pos_y)
    cos_theta_src = dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :]

    h1 = _hankel1(1, k * length)
    # Normal derivative of the Green's function: d/dn H0(kr) = -k H1(kr) cos(theta).
    d_h0_dn = -k * h1 * cos_theta_src
    integrand = d_h0_dn * src_field * src_dx
    return -0.5j * jnp.sum(integrand, axis=1)


@jax.jit
def _hypersingular_kernel(
    k,
    src_field,
    src_pos_x,
    src_pos_y,
    src_normal_x,
    src_normal_y,
    src_dx,
    dst_pos_x,
    dst_pos_y,
    dst_normal_x,
    dst_normal_y,
):
    """Hypersingular operator, needed for a Dual BEM over zero-thickness objects.

    UNVALIDATED. The 1/r^2 singularity when a source and destination element
    coincide is not treated; doing so needs singularity subtraction or a
    Galerkin weak form.
    """
    length, dir_x, dir_y = _pair_geometry(src_pos_x, src_pos_y, dst_pos_x, dst_pos_y)

    cos_theta_src = dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :]
    cos_theta_dst = dir_x * dst_normal_x[:, jnp.newaxis] + dir_y * dst_normal_y[:, jnp.newaxis]
    # Destination normals vary along axis 0, so they index as a column.
    cos_theta_normals = (
        src_normal_x[jnp.newaxis, :] * dst_normal_x[:, jnp.newaxis]
        + src_normal_y[jnp.newaxis, :] * dst_normal_y[:, jnp.newaxis]
    )

    kl = k * length
    integrand = _hankel1(2, kl) * cos_theta_src * cos_theta_dst - _hankel1(1, kl) * cos_theta_normals / kl
    integrand = integrand * k**2 * src_field * src_dx
    return 0.25j * jnp.sum(integrand, axis=1)


def _propagate_chunked(kernel, k, src_field, src_obj, dst_obj, max_matrix_entries):
    """Apply `kernel` over destination chunks to bound peak memory.

    The chunk loop runs on the host rather than through `lax.map` because the
    kernels call `pure_callback`, which does not vectorise cleanly under vmap.
    Only two distinct chunk shapes occur, so this costs at most two compiles.
    """
    n_src = src_obj["pos_x"].shape[0]
    n_dst = dst_obj["pos_x"].shape[0]
    chunk = chunk_size_for(n_src, n_dst, max_matrix_entries)

    pieces = []
    for start in range(0, n_dst, chunk):
        end = min(start + chunk, n_dst)
        pieces.append(
            kernel(
                k,
                src_field,
                src_obj["pos_x"],
                src_obj["pos_y"],
                src_obj["normal_x"],
                src_obj["normal_y"],
                src_obj["dx"],
                dst_obj["pos_x"][start:end],
                dst_obj["pos_y"][start:end],
            )
        )

    if len(pieces) > 1:
        logger.debug("propagated in %d chunks of up to %d destinations", len(pieces), chunk)
    return jnp.concatenate(pieces) if len(pieces) > 1 else pieces[0]


def propagate_mirror(
    k,
    src_field: jax.Array,
    src_obj: ObjectSamples,
    dst_obj: ObjectSamples,
    max_matrix_entries: int = DEFAULT_MAX_MATRIX_ENTRIES,
) -> jax.Array:
    """Single-layer (reflecting) near-field propagation. Conforms to `Propagator`."""
    return _propagate_chunked(_mirror_kernel, k, src_field, src_obj, dst_obj, max_matrix_entries)


def propagate_slit(
    k,
    src_field: jax.Array,
    src_obj: ObjectSamples,
    dst_obj: ObjectSamples,
    max_matrix_entries: int = DEFAULT_MAX_MATRIX_ENTRIES,
) -> jax.Array:
    """Double-layer (transmitting) near-field propagation. Conforms to `Propagator`."""
    return _propagate_chunked(_slit_kernel, k, src_field, src_obj, dst_obj, max_matrix_entries)
