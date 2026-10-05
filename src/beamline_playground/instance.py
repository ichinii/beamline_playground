import math
import numpy as np
import jax.numpy as jnp
from .scene import Scene, Object, Source, Mirror, Slit, Detector, Segment, Circle, Arc, Ellipse, Parabola

def _distance(a, b):
    return np.sqrt(np.sum((a - b)**2))

# TODO: put all geometry instantiation functions into a kernel and run it before simulate

def _instantiate_segment(geometry, samples_per_wavelength, wavelength):
    pos_a = np.array([geometry.pos_a.x, geometry.pos_a.y])
    pos_b = np.array([geometry.pos_b.x, geometry.pos_b.y])
    l = _distance(pos_a, pos_b)
    n = math.ceil(l * samples_per_wavelength / wavelength)
    pos_x = jnp.linspace(pos_a[0], pos_b[0], n)
    pos_y = jnp.linspace(pos_a[1], pos_b[1], n)
    normal_x = jnp.full((n,), (pos_b[1] - pos_a[1]) / l)
    normal_y = jnp.full((n,), (pos_a[0] - pos_b[0]) / l)
    dx = jnp.full((n,), l / n)
    return pos_x, pos_y, normal_x, normal_y, dx

def _instantiate_circle(geometry, samples_per_wavelength, wavelength):
    pos = np.array([geometry.pos.x, geometry.pos.y])
    radius = geometry.radius
    circumference = 2 * math.pi * radius
    n = math.ceil(circumference * samples_per_wavelength / wavelength)
    angles = jnp.linspace(0, 2 * math.pi, n)
    pos_x = pos[0] + radius * jnp.cos(angles)
    pos_y = pos[1] + radius * jnp.sin(angles)
    normal_x = jnp.cos(angles) if geometry.normal_inward else -jnp.cos(angles)
    normal_y = jnp.sin(angles) if geometry.normal_inward else -jnp.sin(angles)
    dx = jnp.full((n,), circumference / n)
    return pos_x, pos_y, normal_x, normal_y, dx

def _instantiate_arc(geometry, samples_per_wavelength, wavelength):
    pos = np.array([geometry.pos.x, geometry.pos.y])
    radius = geometry.radius
    angle_range = geometry.angle
    arc_length = radius * angle_range
    n = math.ceil(arc_length * samples_per_wavelength / wavelength)
    angles = jnp.linspace(-angle_range/2, angle_range/2, n)
    pos_x = pos[0] + radius * jnp.cos(angles)
    pos_y = pos[1] + radius * jnp.sin(angles)
    normal_x = jnp.cos(angles) if geometry.normal_inward else -jnp.cos(angles)
    normal_y = jnp.sin(angles) if geometry.normal_inward else -jnp.sin(angles)
    dx = jnp.full((n,), arc_length / n)
    return pos_x, pos_y, normal_x, normal_y, dx

def _instantiate_ellipse(geometry, samples_per_wavelength, wavelength):
    pos = np.array([geometry.pos.x, geometry.pos.y])
    radius_x = geometry.radius_x
    radius_y = geometry.radius_y
    angle_range = geometry.angle
    # Approximate length of the elliptical segment using numerical integration
    angles = jnp.linspace(-angle_range/2, angle_range/2, 100)
    x = pos[0] + radius_x * jnp.cos(angles)
    y = pos[1] + radius_y * jnp.sin(angles)
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    segment_length = jnp.sum(jnp.sqrt(dx**2 + dy**2))
    n = math.ceil(segment_length * samples_per_wavelength / wavelength)
    angles_n = jnp.linspace(-angle_range/2, angle_range/2, n)
    pos_x = pos[0] + radius_x * jnp.cos(angles_n)
    pos_y = pos[1] + radius_y * jnp.sin(angles_n)
    normal_x = (radius_y * jnp.cos(angles_n)) if geometry.normal_inward else (-radius_y * jnp.cos(angles_n))
    normal_y = (radius_x * jnp.sin(angles_n)) if geometry.normal_inward else (-radius_x * jnp.sin(angles_n))
    dx_n = segment_length / n
    dx_array = jnp.full((n,), dx_n)
    return pos_x, pos_y, normal_x, normal_y, dx_array

def _instantiate_parabola(geometry, samples_per_wavelength, wavelength):
    pos = np.array([geometry.pos.x, geometry.pos.y])
    focal_length = geometry.focal_length
    angle_range = geometry.angle
    # Approximate length of the parabolic segment using numerical integration
    angles = jnp.linspace(-angle_range/2, angle_range/2, 100)
    x = pos[0] + angles
    y = pos[1] + (x - pos[0])**2 / (4 * focal_length)
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    segment_length = jnp.sum(jnp.sqrt(dx**2 + dy**2))
    n = math.ceil(segment_length * samples_per_wavelength / wavelength)
    angles_n = jnp.linspace(-angle_range/2, angle_range/2, n)
    pos_x = pos[0] + angles_n
    pos_y = pos[1] + (pos_x - pos[0])**2 / (4 * focal_length)
    normal_x = jnp.full((n,), 0.0)  # Placeholder for normal vector calculation
    normal_y = jnp.full((n,), 1.0)  # Placeholder for normal vector calculation
    dx_n = segment_length / n
    dx_array = jnp.full((n,), dx_n)
    return pos_x, pos_y, normal_x, normal_y, dx_array

def _instantiate_geometry(geometry, samples_per_wavelength, wavelength):
    if isinstance(geometry, Segment):
        return _instantiate_segment(geometry, samples_per_wavelength, wavelength)
    elif isinstance(geometry, Circle):
        return _instantiate_circle(geometry, samples_per_wavelength, wavelength)
    elif isinstance(geometry, Arc):
        return _instantiate_arc(geometry, samples_per_wavelength, wavelength)
    elif isinstance(geometry, Ellipse):
        return _instantiate_ellipse(geometry, samples_per_wavelength, wavelength)
    elif isinstance(geometry, Parabola):
        return _instantiate_parabola(geometry, samples_per_wavelength, wavelength)
    else:
        raise ValueError(f"Unsupported geometry type: {type(geometry)}")

def _instantiate_object(obj, samples_per_wavelength, wavelength):
    pos_x, pos_y, normal_x, normal_y, dx = _instantiate_geometry(obj.geometry, samples_per_wavelength, wavelength)
    return {
        "pos_x": pos_x,
        "pos_y": pos_y,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "dx": dx,
    }

class SceneInstance():
    def __init__(self, scene: Scene):
        self.name = scene.name
        self.wavelength = scene.wavelength
        self.samples_per_wavelength = scene.samples_per_wavelength
        self.objs = [_instantiate_object(obj, scene.samples_per_wavelength, scene.wavelength) for obj in scene.objs]
        self.dag = tuple(tuple(child_indices) for child_indices in scene.dag)
