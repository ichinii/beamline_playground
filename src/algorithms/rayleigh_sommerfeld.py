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
    @param dst_pos_x: x coordinates of samples in the destination object
    @param dst_pos_y: y coordinates of samples in the destination object
    @param wavelength: wavelength of the wave
"""
def _rayleigh_sommerfeld_kernel(
    k,
    src_field,
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx,
    dst_pos_x, dst_pos_y,
):
    # cartesian product of the samples in the source and destination objects
    # shapes: (num_dst, 1) and (1, num_src)
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    # 2. Distance and direction matrices
    vel_x = bx - ax
    vel_y = by - ay
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    # RS1 obliquity factor: cos(theta) where theta is the angle between the normal of the source and the direction to the destination
    # this is Rayleigh-Sommerfeld correction to the Huygens-Fresnel principle,
    # which accounts for the fact that the contribution from each sample in the source is not isotropic,
    # but depends on the angle to the destination
    # ensures that the total power is conserved
    cos_theta_line = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])

    # check if source is a single point using static shapes (evaluates at compile time)
    if src_pos_x.shape[0] == 1:
        cos_theta = jnp.ones_like(cos_theta_line)
    else:
        cos_theta = cos_theta_line

    # shape: (num_dst, num_src)
    p = (jnp.exp(1j * k * l) / jnp.sqrt(l)) * cos_theta * src_field

    # riemann sum: sum over source samples of p * dx, where dx is the spacing between samples in the source object
    dst_field = jnp.sum(p, axis=1) * src_dx

    # normalization factor
    # ensures that the total power is conserved
    # note: 1/sqrt(1j*lambda) == sqrt(k/(2j*pi))
    wavelength = 2.0 * jnp.pi / k
    norm_factor = 1.0 / jnp.sqrt(1j * wavelength)
    dst_field = norm_factor * dst_field

    return dst_field

def _hankel_transmissive_grating_kernel():
    pass

def propagate(k, src_field, src_obj, dst_obj):
    return _rayleigh_sommerfeld_kernel(
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
