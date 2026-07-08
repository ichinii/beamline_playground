import numpy as np
import math
import jax
import jax.numpy as jnp
from enum import Enum

### scene ###

class Scene:
    def __init__(self, name, samples_per_wavelength, wavelength):
        self.name = name
        self.samples_per_wavelength = samples_per_wavelength
        self.wavelength = wavelength
        self.objs = []

    def append_point(self, pos):
        self.objs.append({
            "geometry": "point",
            "pos": pos,
        })

    # def append_dipole(self, pos, normal):
    #     self.objs.append({
    #         "geometry": "point",
    #         "pos": pos,
    #         "normal": normal,
    #     })

    # def append_knife(self, pos, normal):
    #     self.objs.append({
    #         "geometry": "knife",
    #         "pos": pos,
    #         "normal": normal,
    #     })

    def append_line(self, pos_a, pos_b):
        self.objs.append({
            "geometry": "line",
            "pos_a": pos_a,
            "pos_b": pos_b,
        })

    def append_slit(self, pos_a, pos_b):
        self.objs.append({
            "slit": True,
            "geometry": "line",
            "pos_a": pos_a,
            "pos_b": pos_b,
        })

    def append_circle(self, pos, radius, normal_inward=False):
        self.objs.append({
            "geometry": "circle",
            "pos": pos,
            "radius": radius,
            "normal_inward": normal_inward,
        })

    def append_arc(self, pos, radius, angle_start, angle_end, normal_inward=False):
        self.objs.append({
            "geometry": "arc",
            "pos": pos,
            "radius": radius,
            "angle_start": angle_start,
            "angle_end": angle_end,
            "normal_inward": normal_inward,
        })

    def instantiate(self):
        def instantiate_object(obj):
            return _instantiate_object(obj, self.samples_per_wavelength, self.wavelength)

        objs = [instantiate_object(obj) for obj in self.objs]

        return SceneInstance(self.wavelength, objs)

class SceneInstance:
    def __init__(self, wavelength, objs):
        self.wavelength = wavelength
        self.objs = objs

def _instantiate_object(obj, samples_per_wavelength, wavelength):
    sample_density = samples_per_wavelength / wavelength

    def instantiate_point(obj):
        return {
            "pos_x": jnp.array([obj["pos"][0]]),
            "pos_y": jnp.array([obj["pos"][1]]),
            "normal_x": jnp.array([1.0]),
            "normal_y": jnp.array([0.0]),
            "dx": jnp.array([1.0]),
        }

    # def instantiate_dipole(obj):
    #     normal = np.array(obj["normal"])
    #     normal /= np.linalg.norm(normal)
    #     return {
    #         "pos_x": jnp.array([obj["pos"][0]]),
    #         "pos_y": jnp.array([obj["pos"][1]]),
    #         "normal_x": jnp.array([normal[0]]),
    #         "normal_y": jnp.array([normal[1]]),
    #         "dx": jnp.array([1.0]),
    #     }

    def instantiate_line(obj):
        pos_a = np.array(obj["pos_a"])
        pos_b = np.array(obj["pos_b"])
        l = np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
        n = math.ceil(l * sample_density)
        v = pos_b - pos_a
        return {
            "pos_x": jnp.linspace(pos_a[0], pos_b[0], n),
            "pos_y": jnp.linspace(pos_a[1], pos_b[1], n),
            "normal_x": jnp.array([v[1] / l]),
            "normal_y": jnp.array([v[0] / -l]),
            "dx": jnp.array([l / (n - 1)]),
        }

    def instantiate_circle(obj):
        pos = np.array(obj["pos"])
        radius = obj["radius"]
        normal_inward = obj["normal_inward"]

        circumference = 2 * math.pi * radius
        n = math.ceil(circumference * sample_density)
        angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        normal_x = jnp.cos(angles)
        normal_y = jnp.sin(angles)
        pos_x = pos[0] + radius * normal_x
        pos_y = pos[1] + radius * normal_y
        normal_x = -normal_x if normal_inward else normal_x
        normal_y = -normal_y if normal_inward else normal_y
        dx = circumference / n
        return {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "dx": jnp.array([dx]),
        }

    def instantiate_arc(obj):
        pos = np.array(obj["pos"])
        radius = obj["radius"]
        angle_start = obj["angle_start"]
        angle_end = obj["angle_end"]
        normal_inward = obj["normal_inward"]

        arc_length = radius * (angle_end - angle_start)
        n = math.ceil(arc_length * sample_density)
        angles = jnp.linspace(angle_start, angle_end, n)
        normal_x = jnp.cos(angles)
        normal_y = jnp.sin(angles)
        pos_x = pos[0] + normal_x * radius
        pos_y = pos[1] + normal_y * radius
        normal_x = -normal_x if normal_inward else normal_x
        normal_y = -normal_y if normal_inward else normal_y
        dx = arc_length / n
        return {
            "pos_x": pos_x,
            "pos_y": pos_y,
            "normal_x": normal_x,
            "normal_y": normal_y,
            "dx": jnp.array([dx]),
        }

    if obj["geometry"] == "point": return instantiate_point(obj)
    # elif obj["geometry"] == "dipole": return instantiate_dipole(obj)
    elif obj["geometry"] == "line": return instantiate_line(obj)
    elif obj["geometry"] == "circle": return instantiate_circle(obj)
    elif obj["geometry"] == "arc": return instantiate_arc(obj)
    else:
        raise ValueError(f"unknown geometry: {obj["geometry"]}")
