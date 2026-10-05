"""Run a scene: propagate fields along the DAG and report the result.

Which kernel is used for a propagation step depends on the *source* object's
type, so a mirror reflects, a slit transmits, and a detector -- being a
terminal sink -- never re-radiates at all.

Two kernel maps are provided. `FAR_FIELD_PROPAGATORS` is the default and uses
the Rayleigh-Sommerfeld integral throughout; it is the validated path.
`NEAR_FIELD_PROPAGATORS` selects the Hankel boundary-element kernels, which
are valid close to an object but are still experimental.
"""

import logging
from collections.abc import Mapping
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field

from .algorithms import Propagator, hankel_mirror, hankel_slit, rayleigh_sommerfeld
from .instance import SceneInstance
from .scene import Scene

logger = logging.getLogger(__name__)

ObjectType = Literal["source", "detector", "mirror", "slit"]
PropagatorMap = Mapping[str, Propagator]

#: Far-field kernels (default). Accurate when propagation distances are large
#: compared to the wavelength.
FAR_FIELD_PROPAGATORS: PropagatorMap = {
    "source": rayleigh_sommerfeld,
    "mirror": rayleigh_sommerfeld,
    "slit": rayleigh_sommerfeld,
}

#: Near-field boundary-element kernels. Experimental -- see `algorithms.hankel`.
NEAR_FIELD_PROPAGATORS: PropagatorMap = {
    "source": hankel_slit,
    "mirror": hankel_mirror,
    "slit": hankel_slit,
}


class ObjectResult(BaseModel):
    id: str = Field(
        description="Unique identifier for the object. Matches the id of the corresponding object in the scene."
    )
    name: str = Field(description="Name for the object. Matches the name of the corresponding object in the scene.")
    type: ObjectType = Field(
        description="Type of the object. Matches the type of the corresponding object in the scene."
    )
    sample_x: list[float] = Field(
        description="Sample positions along the object, measured as arc length from its midpoint, in millimeters."
    )
    field_re: list[float] = Field(description="Real part of the complex field at each sample position.")
    field_im: list[float] = Field(description="Imaginary part of the complex field at each sample position.")
    intensity_y: list[float] = Field(description="Intensity (squared field magnitude) at each sample position.")

    @property
    def field(self) -> np.ndarray:
        """The complex field, reassembled as a numpy array."""
        return np.asarray(self.field_re) + 1j * np.asarray(self.field_im)


class SimulationResult(BaseModel):
    scene_name: str = Field(description="Name of the scene that was simulated.")
    objs: list[ObjectResult] = Field(description="List of results for each object in the scene.")

    def by_id(self, obj_id: str) -> ObjectResult:
        for obj in self.objs:
            if obj.id == obj_id:
                return obj
        raise KeyError(obj_id)


@partial(jax.jit, static_argnames=["dag", "order", "is_source", "propagators"])
def _simulate_impl(k, objs, dag, order, is_source, propagators):
    """Accumulate fields across the DAG.

    `propagators[i]` is the kernel used when object i radiates; it is None for
    objects that never radiate (detectors).

    `order` is a topological ordering of the objects. Propagating in list
    order instead would evaluate an object before its illuminator had been
    computed, silently yielding zeros.

    `dag`, `order`, `is_source` and `propagators` are static because the Python
    loop below is unrolled at trace time into a fixed XLA graph; their values
    decide the graph's shape, so they must be known when tracing.

    `k` is traced rather than static, which avoids a recompile when the
    wavelength changes without changing any element count. Note that a
    wavelength change large enough to change `sample_count` alters the array
    shapes, and jit retraces on shapes regardless.
    """

    def initial_field(index):
        n = objs[index]["pos_x"].shape[0]
        # Sources emit at unit amplitude; everything else starts dark and is
        # filled in by its dependencies.
        value = 1.0 + 0.0j if is_source[index] else 0.0 + 0.0j
        return jnp.full((n,), value, dtype=jnp.complex64)

    fields = [initial_field(i) for i in range(len(dag))]

    for dst_index in order:
        for src_index in dag[dst_index]:
            propagate = propagators[src_index]
            fields[dst_index] = fields[dst_index] + propagate(k, fields[src_index], objs[src_index], objs[dst_index])

    return fields


def simulate(scene: Scene, propagators: PropagatorMap = FAR_FIELD_PROPAGATORS) -> SimulationResult:
    """Simulate `scene` and return the field and intensity along every object.

    Args:
        scene: the scene to simulate.
        propagators: kernel to use per radiating object type. Defaults to
            `FAR_FIELD_PROPAGATORS`.
    """
    instance = SceneInstance(scene)
    k = 2.0 * jnp.pi / scene.wavelength

    types = [obj.type for obj in scene.objs]
    is_source = tuple(t == "source" for t in types)

    # Resolve one kernel per object up front; detectors get None because the
    # scene validator guarantees nothing depends on them.
    per_object: list[Propagator | None] = []
    for index, obj_type in enumerate(types):
        if obj_type == "detector":
            per_object.append(None)
            continue
        if obj_type not in propagators:
            raise ValueError(
                f"object {index} has type {obj_type!r}, for which no propagator "
                f"was supplied (have: {sorted(propagators)})"
            )
        per_object.append(propagators[obj_type])

    logger.debug(
        "simulating %r: %d objects, %d total elements",
        scene.name,
        len(scene.objs),
        scene.total_sample_count(),
    )

    fields = _simulate_impl(k, instance.objs, instance.dag, instance.order, is_source, tuple(per_object))

    def object_result(index: int) -> ObjectResult:
        obj = scene.objs[index]
        field = np.asarray(fields[index])
        return ObjectResult(
            id=obj.id,
            name=obj.name,
            type=obj.type,
            # Arc-length coordinates come from the discretisation itself, so
            # they always line up with the field samples.
            sample_x=instance.sample_coords[index].tolist(),
            field_re=field.real.tolist(),
            field_im=field.imag.tolist(),
            intensity_y=(np.abs(field) ** 2).tolist(),
        )

    return SimulationResult(
        scene_name=scene.name,
        objs=[object_result(i) for i in range(len(scene.objs))],
    )
