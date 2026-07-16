import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from concurrent.futures import ThreadPoolExecutor
import os
from functools import partial

_NUM_CPUS = os.cpu_count() or 4
_POOL = ThreadPoolExecutor(max_workers=_NUM_CPUS)

def _cpu_parallel(fn, x):
    orig_shape = x.shape

    out_dtype = np.complex64 if x.dtype == np.float32 else np.complex128
    out = np.empty(orig_shape, dtype=out_dtype)

    x_flat = x.reshape(-1)
    out_flat = out.reshape(-1)
    n = len(x_flat)

    if n < 2048:
        fn(x, out=out)
        return out

    chunk_size = (n + _NUM_CPUS - 1)
    futures = []

    for i in range(_NUM_CPUS):
        start = i * chunk_size
        end = min(start + chunk_size, n)
        if start < end:
            futures.append(
                _POOL.submit(fn, x_flat[start:end], out=out_flat[start:end])
            )

    # wait for all CPU threads to finish
    for future in futures:
        future.result()

    return out

def _jax_h(order, kr):
    result_shape = jax.ShapeDtypeStruct(
        kr.shape,
        jnp.complex64 if kr.dtype == jnp.float32 else jnp.complex128
    )
    return jax.pure_callback(
        partial(_cpu_parallel, partial(scipy.special.hankel1, order)),
        result_shape,
        kr,
    )

@jax.jit
def _jax_h0(kr):
    return _jax_h(0, kr)

@jax.jit
def _jax_h1(kr):
    return _jax_h(1, kr)

@jax.jit(static_argnames=["src_slit", "dst_slit"])
def _test_thin_objects_compat(
    src_pos_x, src_pos_y,
    dst_pos_x, dst_pos_y,
    src_normal_x, src_normal_y,
    dst_normal_x, dst_normal_y,
    src_slit,
    dst_slit,
):
    vel_x = dst_pos_x[:, jnp.newaxis] - src_pos_x[jnp.newaxis, :]
    vel_y = dst_pos_y[:, jnp.newaxis] - src_pos_y[jnp.newaxis, :]
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    cos_theta_src = dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :]
    cos_theta_dst = dir_x * dst_normal_x[:, jnp.newaxis] + dir_y * dst_normal_y[:, jnp.newaxis]
    cos_theta_src = jnp.sign(cos_theta_src)
    cos_theta_dst = jnp.sign(cos_theta_dst)

    src_sum = jnp.sum(cos_theta_src)
    # src_sum = (src_sum if src_slit else jnp.abs(src_sum)) # allow mirrors radiate in both directions

    dst_sum = jnp.sum(cos_theta_dst)
    dst_sum = (dst_sum if dst_slit else -dst_sum) # ensure mirrors face towards the source

    n = src_pos_x.shape[0] * dst_pos_x.shape[0]
    src_ok = src_sum == n
    dst_ok = dst_sum == n

    return src_ok, dst_ok

@jax.jit
def _hankel_mirror_kernel(
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx, src_field,
    dst_pos_x, dst_pos_y,
    wavelength
):
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    vel_x = bx - ax
    vel_y = by - ay
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    cos_theta_src = jnp.abs(dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :])

    k = 2.0 * jnp.pi / wavelength
    H0_matrix = _jax_h0(k * l)
    dst_field_matrix = H0_matrix * src_field * cos_theta_src
    dst_field = jnp.sum(dst_field_matrix, axis=1) * src_dx
    dst_field = -0.5 * dst_field * k
    return dst_field

@jax.jit
def _hankel_slit_kernel(
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx, src_field,
    dst_pos_x, dst_pos_y,
    wavelength
):
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    vel_x = bx - ax
    vel_y = by - ay
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    cos_theta_src = dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :]

    k = 2.0 * jnp.pi / wavelength
    H1_matrix = _jax_h1(k * l)
    dH0_dx = -k * H1_matrix * cos_theta_src
    p = dH0_dx * src_field
    dst_field = jnp.sum(p, axis=1) * src_dx
    dst_field = -0.5j * dst_field # original
    # dst_field = 0.5j * dst_field # idk
    return dst_field

@jax.jit
def _hankel_transmissive_grating_kernel():
    pass

@jax.jit
def _hankel_reflective_grating_kernel():
    pass

# requied for Dual Boundary Element Method (when objects have no thickness)
# has the problem of 1/(r^2) singularity when source and destination are at the same position
# -> Singularity Subtraction or switch to a Galerkin weak-form integration
@jax.jit
def _hankel_hypersingular_kernel(
    src_pos_x, src_pos_y, src_normal_x, src_normal_y, src_dx, src_field,
    dst_pos_x, dst_pos_y, dst_normal_x, dst_normal_y, dst_dx,
    wavelength
):
    ax = src_pos_x[jnp.newaxis, :]
    bx = dst_pos_x[:, jnp.newaxis]
    ay = src_pos_y[jnp.newaxis, :]
    by = dst_pos_y[:, jnp.newaxis]

    vel_x = bx - ax
    vel_y = by - ay
    l = jnp.sqrt(vel_x**2 + vel_y**2 + 1e-12)
    dir_x = vel_x / l
    dir_y = vel_y / l

    cos_theta_src = dir_x * src_normal_x[jnp.newaxis, :] + dir_y * src_normal_y[jnp.newaxis, :]
    cos_theta_dst = dir_x * dst_normal_x[:, jnp.newaxis] + dir_y * dst_normal_y[:, jnp.newaxis]
    cos_theta_normals = src_normal_x[jnp.newaxis, :] * dst_normal_x[jnp.newaxis, :] + src_normal_y[jnp.newaxis, :] * dst_normal_y[jnp.newaxis, :]

    k = 2.0 * jnp.pi / wavelength
    kl = k * l
    dst_field_matrix = _jax_h(2, kl) * cos_theta_src * cos_theta_dst - _jax_h(1, kl) * cos_theta_normals / kl
    dst_field_matrix *= k**2
    dst_field_matrix *= src_field
    dst_field = jnp.sum(dst_field_matrix, axis=1) * src_dx
    dst_field *= 0.25j
    return dst_field

def propagate(d_scene, ia, ib, d_src_field, src_slit, dst_slit):
    src_ok, dst_ok = _test_thin_objects_compat(
        d_scene.objs[ia]["pos_x"],
        d_scene.objs[ia]["pos_y"],
        d_scene.objs[ib]["pos_x"],
        d_scene.objs[ib]["pos_y"],
        d_scene.objs[ia]["normal_x"],
        d_scene.objs[ia]["normal_y"],
        d_scene.objs[ib]["normal_x"],
        d_scene.objs[ib]["normal_y"],
        src_slit,
        dst_slit,
    )

    print(f"thin object compatibility check: source ({ia}) ok: {src_ok}, destination ({ib}) ok: {dst_ok}")
    if not src_ok or not dst_ok:
        raise ValueError(f"source and destination objects ({ia} and {ib}) are not compatible for thin object propagation")

    kernel = _hankel_slit_kernel if src_slit else _hankel_mirror_kernel
    print("slit" if src_slit else "mirror", "kernel selected for propagation")

    return kernel(
        d_scene.objs[ia]["pos_x"],
        d_scene.objs[ia]["pos_y"],
        d_scene.objs[ia]["normal_x"],
        d_scene.objs[ia]["normal_y"],
        d_scene.objs[ia]["dx"],
        d_src_field,
        d_scene.objs[ib]["pos_x"],
        d_scene.objs[ib]["pos_y"],
        d_scene.wavelength,
    )
