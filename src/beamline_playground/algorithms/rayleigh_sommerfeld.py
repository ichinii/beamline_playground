import math
import jax
import jax.numpy as jnp
import numpy as np

"""
    2D Rayleigh-Sommerfeld diffraction integral (RS1)
    far-field approximation, because we are missing the Henkel function term that accounts for near-field effects

    @param src_pos_x: x coordinates of samples in the source object
    @param src_pos_y: y coordinates of samples in the source object
    @param src_normal_x: x component of the normal vector of the source object (for lines), or None for points
    @param src_normal_y: y component of the normal vector of the source object (for lines), or None for points
    @param src_dx: spacing between samples in the source object (for lines), or 1 for points
    @param src_field: complex amplitude of the wave at each sample in the source object
    @param dst_pos_x: single x coordinate of sample in the destination object
    @param dst_pos_y: single y coordinate of sample in the destination object
    @param wavelength: wavelength of the wave
"""
def _rayleigh_sommerfeld_kernel(
    k,
    src_field,
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx,
    dst_pos_x, dst_pos_y,
):
    n_src = src_pos_x.shape[0]

    def accumulate(i, acc_dst_sample):
        dx = dst_pos_x - src_pos_x[i]
        dy = dst_pos_y - src_pos_y[i]
        r = jnp.sqrt(dx**2 + dy**2)

        # RS1 obliquity factor
        cos_theta = jnp.abs(dx/r * src_normal_x + dy/r * src_normal_y)
        # geometric attentuation
        att = 1.0/jnp.sqrt(r)
        # per-source-sample contribution to dst sample
        contrib = jnp.exp(1j * k * r) * att * cos_theta * src_field[i] * src_dx[i]
        # accumulation into dst sample
        return acc_dst_sample + contrib.squeeze()

    dst_sample = jax.lax.fori_loop(0, n_src, accumulate, 0.0 + 0.0j)

    # RS1 normalization factor
    # note: 1/sqrt(1j*lambda) == sqrt(k/(2j*pi))
    wavelength = 2.0 * jnp.pi / k
    norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
    dst_sample = norm_factor * dst_sample

    return dst_sample

def _rayleigh_sommerfeld_kernel_simple(
    k,
    src_field,
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx,
    dst_pos_x, dst_pos_y,
):
    dx = dst_pos_x[:, jnp.newaxis] - src_pos_x[jnp.newaxis, :]
    dy = dst_pos_y[:, jnp.newaxis] - src_pos_y[jnp.newaxis, :]

    r = jnp.sqrt(dx**2 + dy**2)
    attenuation = 1.0/jnp.sqrt(r)
    cos_theta = jnp.abs(dx/r * src_normal_x + dy/r * src_normal_y)

    dst_matrix = jnp.exp(1j * k * r) * attenuation * cos_theta * src_field * src_dx

    wavelength = 2.0 * jnp.pi / k
    norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
    dst_field = norm_factor * dst_matrix.sum(axis=1)

    return dst_field

def propagate(k, src_field, src_obj, dst_obj):
    # kernel = jax.vmap(_rayleigh_sommerfeld_kernel, in_axes=(
    #     None,
    #     None,
    #     None, None, None, None, None,
    #     0, 0
    # ))

    return _rayleigh_sommerfeld_kernel_simple(
        k,
        src_field,
        src_obj["pos_x"],
        src_obj["pos_y"],
        src_obj["normal_x"],
        src_obj["normal_y"],
        src_obj["dx"],
        dst_obj["pos_x"],
        dst_obj["pos_y"],
    )
