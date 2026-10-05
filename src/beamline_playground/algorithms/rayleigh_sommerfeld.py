"""2D Rayleigh-Sommerfeld diffraction integral (RS1).

This is a far-field form: it omits the Hankel-function term that accounts for
near-field behaviour, so it is accurate when the propagation distance is large
compared to the wavelength. See `hankel.py` for the near-field kernels.

The destination axis is processed in chunks so that peak memory stays bounded
no matter how finely the source is sampled -- a full n_dst x n_src matrix is
never materialised.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .base import DEFAULT_MAX_MATRIX_ENTRIES, ObjectSamples, chunk_size_for


@partial(jax.jit, static_argnames=["chunk_size"])
def _propagate_impl(
    k,
    src_field,
    src_pos_x,
    src_pos_y,
    src_normal_x,
    src_normal_y,
    src_dx,
    dst_pos_x,
    dst_pos_y,
    chunk_size,
):
    def one_destination(dst):
        dx = dst[0] - src_pos_x
        dy = dst[1] - src_pos_y
        r = jnp.sqrt(dx * dx + dy * dy)

        # RS1 obliquity factor: how obliquely the source element radiates
        # toward this destination point.
        cos_theta = jnp.abs((dx * src_normal_x + dy * src_normal_y) / r)
        # Geometric attenuation in 2D.
        attenuation = 1.0 / jnp.sqrt(r)

        contributions = jnp.exp(1j * k * r) * attenuation * cos_theta * src_field * src_dx
        return jnp.sum(contributions)

    destinations = jnp.stack([dst_pos_x, dst_pos_y], axis=1)
    # batch_size turns this into a chunked vmap: vectorised within a chunk,
    # sequential across chunks.
    dst_field = jax.lax.map(one_destination, destinations, batch_size=chunk_size)

    # RS1 normalisation. Note 1/sqrt(1j*lambda) == sqrt(k/(2j*pi)).
    wavelength = 2.0 * jnp.pi / k
    return dst_field / jnp.sqrt(1j * wavelength)


def propagate(
    k,
    src_field: jax.Array,
    src_obj: ObjectSamples,
    dst_obj: ObjectSamples,
    max_matrix_entries: int = DEFAULT_MAX_MATRIX_ENTRIES,
) -> jax.Array:
    """Propagate `src_field` from `src_obj` to `dst_obj`. Conforms to `Propagator`."""
    n_src = src_obj["pos_x"].shape[0]
    n_dst = dst_obj["pos_x"].shape[0]

    return _propagate_impl(
        k,
        src_field,
        src_obj["pos_x"],
        src_obj["pos_y"],
        src_obj["normal_x"],
        src_obj["normal_y"],
        src_obj["dx"],
        dst_obj["pos_x"],
        dst_obj["pos_y"],
        chunk_size=chunk_size_for(n_src, n_dst, max_matrix_entries),
    )
