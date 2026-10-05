"""Discretised form of a `Scene`, ready to hand to the propagation kernels.

A `Scene` is declarative and resolution independent. A `SceneInstance` pins it
to a concrete sampling: every geometry is turned into boundary elements whose
positions, normals and widths come from `Geometry.sample()`, so this module
contains no geometry maths of its own.
"""

import jax.numpy as jnp

from .scene import Scene


class SceneInstance:
    """A `Scene` discretised at its requested sampling density.

    Attributes:
        objs: one dict per object, holding the jax arrays the kernels consume
            (`pos_x`, `pos_y`, `normal_x`, `normal_y`, `dx`).
        sample_coords: one numpy array per object, giving each sample's
            arc-length coordinate measured from the middle of the geometry.
            Kept out of `objs` so it is never traced into a jitted kernel.
        dag: the scene's dependency graph, as nested tuples so it can be used
            as a static (hashable) jit argument.
    """

    def __init__(self, scene: Scene):
        self.name = scene.name
        self.wavelength = scene.wavelength
        self.samples_per_wavelength = scene.samples_per_wavelength

        self.objs = []
        self.sample_coords = []
        for obj in scene.objs:
            n = obj.geometry.sample_count(scene.samples_per_wavelength, scene.wavelength)
            samples = obj.geometry.sample(n)
            self.objs.append(
                {
                    "pos_x": jnp.asarray(samples.pos_x),
                    "pos_y": jnp.asarray(samples.pos_y),
                    "normal_x": jnp.asarray(samples.normal_x),
                    "normal_y": jnp.asarray(samples.normal_y),
                    "dx": jnp.asarray(samples.dx),
                }
            )
            self.sample_coords.append(samples.s)

        self.dag = tuple(tuple(deps) for deps in scene.dag)

    def __len__(self) -> int:
        return len(self.objs)
