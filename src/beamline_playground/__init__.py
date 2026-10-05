from importlib.metadata import PackageNotFoundError, version

from . import algorithms
from .instance import SceneInstance
from .scene import (
    Arc,
    Circle,
    Detector,
    Ellipse,
    Geometry,
    GeometrySamples,
    Mirror,
    Object,
    Parabola,
    Scene,
    Segment,
    Slit,
    Source,
    Vec2,
    distance,
)
from .simulate import ObjectResult, SimulationResult, simulate

try:
    __version__ = version("beamline-playground")
except PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "algorithms",
    # geometries
    "Vec2",
    "Segment",
    "Circle",
    "Arc",
    "Ellipse",
    "Parabola",
    "GeometrySamples",
    "Geometry",
    "distance",
    # objects
    "Source",
    "Detector",
    "Mirror",
    "Slit",
    "Object",
    # scene & simulation
    "Scene",
    "SceneInstance",
    "simulate",
    "ObjectResult",
    "SimulationResult",
]
