from typing import Literal, List
from pydantic import BaseModel, Field
import numpy as np
import jax
import jax.numpy as jnp
from .algorithms import rayleigh_sommerfeld
from .instance import SceneInstance
from .scene import Scene

# TODO: data should be a numpy array
class ObjectResult(BaseModel):
    id: str = Field(description="Unique identifier for the object. Matches the id of the corresponding object in the scene.")
    name: str = Field(description="Name for the object. Matches the name of the corresponding object in the scene.")
    type: Literal["source", "detector", "mirror", "slit"] = Field(description="Type of the object. Matches the type of the corresponding object in the scene.")
    sample_x: list[float] = Field(description="Sample positions along the object, measured from its midpoint, in millimeters. The array has the same length as field_y.")
    field_y: list[complex] = Field(description="Field values at the corresponding position along the object.")
    intensity_y: list[float] = Field(description="Intensity values at the corresponding position along the object.")

class SimulationResult(BaseModel):
    scene_name: str = Field(description="Name of the scene that was simulated.")
    objs: list[ObjectResult] = Field(description="List of results for each object in the scene.")

def _propagate(k, src_field, src_obj, dst_obj):
    return rayleigh_sommerfeld(k, src_field, src_obj, dst_obj)

@jax.jit(static_argnames=["k", "dag"])
def _simulate_impl(k, objs, dag):
    # init result fields
    def init_field(src_index, is_source):
        n = objs[src_index]["pos_x"].shape[0]
        value = 1.0 + 0.0j if is_source else 0.0 + 0.0j
        return jnp.full((n,), value, dtype=jnp.complex64)
    result_fields = [init_field(i, len(dag[i]) == 0) for i in range(len(dag))]

    # propagate fields through the DAG
    for dst_index, deps in enumerate(dag):
        for src_index in deps:
            src_field = result_fields[src_index]
            src_obj = objs[src_index]
            dst_obj = objs[dst_index]
            result_fields[dst_index] += _propagate(k, src_field, src_obj, dst_obj)

    return result_fields

def simulate(scene: Scene) -> SimulationResult:
    def intensity(field):
        return jnp.abs(field) ** 2

    instance = SceneInstance(scene)
    k = 2.0 * jnp.pi / scene.wavelength
    result_fields = _simulate_impl(k, instance.objs, instance.dag)

    def object_result(i):
        field_y = result_fields[i].tolist()
        intensity_y = intensity(result_fields[i]).tolist()
        num_samples = len(intensity_y)

        obj = scene.objs[i]
        l = obj.geometry.length()
        sample_x = np.linspace(-l/2, l/2, num_samples).tolist()

        return ObjectResult(
            id=obj.id,
            type=obj.type,
            name=obj.name,
            sample_x=sample_x,
            field_y=field_y,
            intensity_y=intensity_y,
        )

    return SimulationResult(
        scene_name=scene.name,
        objs=[object_result(i) for i in range(len(scene.objs))],
    )
