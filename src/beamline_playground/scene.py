"""Declarative description of a beamline scene.

This module owns both the serialisation schema (pydantic models) and the
geometry discretisation, so that arc length and sample positions can never
disagree. Everything downstream (`instance.py`, `simulate.py`) consumes
`Geometry.sample()` rather than re-deriving the maths.

Conventions:
  * All lengths are in millimetres.
  * Sample points sit at *element midpoints*. An object discretised into n
    elements therefore has n samples, each of width `dx`, and `sum(dx)` equals
    the arc length. Closed curves do not double count a shared endpoint.
  * A normal is a unit vector. `normal_inward` selects which of the two
    perpendicular directions is used.
"""

import math
from typing import Annotated, Literal, NamedTuple
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, model_validator

# Resolution used when an arc length has no closed form and must be integrated.
_QUAD_POINTS = 4096


class Vec2(BaseModel):
    x: float = Field(
        description="X coordinate, in millimeters. In an unrotated context, the positive X axis points to the right."
    )
    y: float = Field(
        description="Y coordinate, in millimeters. In an unrotated context, the positive Y axis points upwards."
    )


def distance(a: Vec2, b: Vec2) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


class GeometrySamples(NamedTuple):
    """One geometry discretised into boundary elements.

    All arrays have length n (the number of elements).

    pos_x, pos_y     element midpoint coordinates
    normal_x, ...    unit normal at each midpoint
    dx               element width; sum(dx) == arc length
    s               arc-length coordinate of each midpoint, measured from the
                     middle of the geometry (so it runs from -length/2 to
                     +length/2). This is what a plot's x axis should use.
    """

    pos_x: np.ndarray
    pos_y: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    dx: np.ndarray
    s: np.ndarray


def _midpoints(t0: float, t1: float, n: int) -> np.ndarray:
    """Parameter value at the centre of each of n equal sub-intervals."""
    edges = np.linspace(t0, t1, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _arclength_coord(dx: np.ndarray, length: float) -> np.ndarray:
    """Midpoint arc-length coordinates, centred on the geometry."""
    return np.cumsum(dx) - 0.5 * dx - 0.5 * length


def _unit_normal_from_tangent(tx: np.ndarray, ty: np.ndarray, inward: bool):
    """Rotate a tangent by -90 degrees and normalise.

    For a curve traversed in increasing parameter, (ty, -tx) is the outward
    (convex-side) normal; `inward` flips it.
    """
    nx, ny = ty, -tx
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    if inward:
        nx, ny = -nx, -ny
    return nx, ny


# geometries / curvatures


class _GeometryBase(BaseModel):
    def length(self) -> float:
        raise NotImplementedError

    def sample(self, n: int) -> GeometrySamples:
        raise NotImplementedError

    def sample_count(self, samples_per_wavelength: int, wavelength: float) -> int:
        """Number of elements needed to resolve this geometry."""
        n = math.ceil(self.length() * samples_per_wavelength / wavelength)
        return max(1, int(n))


class Segment(_GeometryBase):
    type: Literal["segment"] = "segment"
    pos_a: Vec2 = Field(
        description="The starting point of the segment, in millimeters. The normal vector is defined as the vector perpendicular to the segment, pointing to the right when looking from pos_a to pos_b."
    )
    pos_b: Vec2 = Field(description="The ending point of the segment in millimeters.")

    def length(self) -> float:
        return distance(self.pos_a, self.pos_b)

    def sample(self, n: int) -> GeometrySamples:
        length = self.length()
        ax, ay = self.pos_a.x, self.pos_a.y
        bx, by = self.pos_b.x, self.pos_b.y
        t = _midpoints(0.0, 1.0, n)
        pos_x = ax + t * (bx - ax)
        pos_y = ay + t * (by - ay)
        # Perpendicular, pointing right when looking from pos_a to pos_b.
        normal_x = np.full(n, (by - ay) / length)
        normal_y = np.full(n, (ax - bx) / length)
        dx = np.full(n, length / n)
        return GeometrySamples(pos_x, pos_y, normal_x, normal_y, dx, _arclength_coord(dx, length))


class Circle(_GeometryBase):
    type: Literal["circle"] = "circle"
    pos: Vec2 = Field(description="The center of the circle, in millimeters.")
    radius: float = Field(gt=0, description="The radius of the circle, in millimeters.")
    normal_inward: bool = Field(
        description="If true, the normal vector points inward toward the center of the circle. If false, the normal vector points outward away from the center of the circle."
    )

    def length(self) -> float:
        return 2.0 * math.pi * self.radius

    def sample(self, n: int) -> GeometrySamples:
        length = self.length()
        # Closed curve: midpoints over [0, 2pi) never repeat a point.
        theta = _midpoints(0.0, 2.0 * math.pi, n)
        pos_x = self.pos.x + self.radius * np.cos(theta)
        pos_y = self.pos.y + self.radius * np.sin(theta)
        # (cos, sin) points away from the centre, i.e. outward.
        sign = -1.0 if self.normal_inward else 1.0
        normal_x = sign * np.cos(theta)
        normal_y = sign * np.sin(theta)
        dx = np.full(n, length / n)
        return GeometrySamples(pos_x, pos_y, normal_x, normal_y, dx, _arclength_coord(dx, length))


class Arc(_GeometryBase):
    type: Literal["arc"] = "arc"
    pos: Vec2 = Field(description="The center of the arc, in millimeters.")
    radius: float = Field(gt=0, description="The radius of the arc, in millimeters.")
    angle: float = Field(
        gt=0,
        le=2.0 * math.pi,
        description="The angular extent of the arc, in radians. The arc is centered on the positive X axis direction from pos, extending by angle/2 in both directions.",
    )
    normal_inward: bool = Field(
        description="If true, the normal vector points inward toward the center of the arc. If false, the normal vector points outward away from the center of the arc."
    )

    def length(self) -> float:
        return self.radius * self.angle

    def sample(self, n: int) -> GeometrySamples:
        length = self.length()
        theta = _midpoints(-self.angle / 2.0, self.angle / 2.0, n)
        pos_x = self.pos.x + self.radius * np.cos(theta)
        pos_y = self.pos.y + self.radius * np.sin(theta)
        sign = -1.0 if self.normal_inward else 1.0
        normal_x = sign * np.cos(theta)
        normal_y = sign * np.sin(theta)
        dx = np.full(n, length / n)
        return GeometrySamples(pos_x, pos_y, normal_x, normal_y, dx, _arclength_coord(dx, length))


class Ellipse(_GeometryBase):
    type: Literal["ellipse"] = "ellipse"
    pos: Vec2 = Field(description="The center of the ellipse, in millimeters.")
    radius_x: float = Field(gt=0, description="The radius of the ellipse along the X axis, in millimeters.")
    radius_y: float = Field(gt=0, description="The radius of the ellipse along the Y axis, in millimeters.")
    angle: float = Field(
        gt=0,
        le=2.0 * math.pi,
        description="The angular extent in the ellipse's parametric angle, in radians. The segment is centered on the positive X axis direction from pos, extending by angle/2 in both directions. Note this is a parametric angle, not a geometric one.",
    )
    normal_inward: bool = Field(
        description="If true, the normal vector points inward toward the center of the ellipse. If false, the normal vector points outward away from the center of the ellipse."
    )

    def _speed(self, theta: np.ndarray) -> np.ndarray:
        """|dp/dtheta| -- the parametrisation is not arc-length uniform."""
        return np.hypot(self.radius_x * np.sin(theta), self.radius_y * np.cos(theta))

    def length(self) -> float:
        theta = _midpoints(-self.angle / 2.0, self.angle / 2.0, _QUAD_POINTS)
        return float(np.sum(self._speed(theta)) * self.angle / _QUAD_POINTS)

    def sample(self, n: int) -> GeometrySamples:
        theta = _midpoints(-self.angle / 2.0, self.angle / 2.0, n)
        d_theta = self.angle / n
        pos_x = self.pos.x + self.radius_x * np.cos(theta)
        pos_y = self.pos.y + self.radius_y * np.sin(theta)
        # Tangent dp/dtheta = (-rx sin, ry cos).
        normal_x, normal_y = _unit_normal_from_tangent(
            -self.radius_x * np.sin(theta),
            self.radius_y * np.cos(theta),
            self.normal_inward,
        )
        # Element width varies along the curve, so dx is not constant.
        dx = self._speed(theta) * d_theta
        return GeometrySamples(pos_x, pos_y, normal_x, normal_y, dx, _arclength_coord(dx, float(np.sum(dx))))


class Parabola(_GeometryBase):
    type: Literal["parabola"] = "parabola"
    pos: Vec2 = Field(description="The vertex of the parabola, in millimeters.")
    focal_length: float = Field(
        gt=0,
        description="The focal length of the parabola, in millimeters. The parabola opens along the positive Y axis.",
    )
    extent: float = Field(
        gt=0,
        description="Width of the parabola along the X axis, in millimeters. The segment spans pos.x - extent/2 to pos.x + extent/2.",
    )
    normal_inward: bool = Field(
        description="If true, the normal vector points inward toward the concave side of the parabola (where the focus lies). If false, it points outward away from the focus."
    )

    def _speed(self, u: np.ndarray) -> np.ndarray:
        """|dp/du| for p(u) = (u, u^2 / 4f)."""
        return np.hypot(1.0, u / (2.0 * self.focal_length))

    def length(self) -> float:
        u = _midpoints(-self.extent / 2.0, self.extent / 2.0, _QUAD_POINTS)
        return float(np.sum(self._speed(u)) * self.extent / _QUAD_POINTS)

    def sample(self, n: int) -> GeometrySamples:
        u = _midpoints(-self.extent / 2.0, self.extent / 2.0, n)
        du = self.extent / n
        pos_x = self.pos.x + u
        pos_y = self.pos.y + u**2 / (4.0 * self.focal_length)
        # Tangent dp/du = (1, u / 2f); the concave side faces +Y.
        normal_x, normal_y = _unit_normal_from_tangent(
            np.ones(n),
            u / (2.0 * self.focal_length),
            self.normal_inward,
        )
        dx = self._speed(u) * du
        return GeometrySamples(pos_x, pos_y, normal_x, normal_y, dx, _arclength_coord(dx, float(np.sum(dx))))


Geometry = Annotated[Segment | Circle | Arc | Ellipse | Parabola, Field(discriminator="type")]

# objects


class ObjectBase(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex, description="Unique identifier for the object.")
    name: str = Field(default="<unnamed object>", description="Name for the object.")


class Source(ObjectBase):
    type: Literal["source"] = "source"
    geometry: Geometry = Field(description="Geometry of the object.")


class Detector(ObjectBase):
    type: Literal["detector"] = "detector"
    geometry: Geometry = Field(description="Geometry of the object.")


class Mirror(ObjectBase):
    type: Literal["mirror"] = "mirror"
    geometry: Geometry = Field(description="Geometry of the object.")


class Slit(ObjectBase):
    type: Literal["slit"] = "slit"
    geometry: Segment = Field(
        description="Geometry of the object. Always a segment, since slits are always straight lines."
    )


Object = Annotated[Source | Detector | Mirror | Slit, Field(discriminator="type")]


class Scene(BaseModel):
    name: str = Field(default="<unnamed scene>", description="Name for the scene.")
    objs: list[Object] = Field(description="List of objects in the scene.")
    dag: list[list[int]] = Field(
        description="Directed acyclic graph (DAG) representing the relationships between objects in the scene. Each index in the outer list corresponds to an object in the objs list, and the inner lists contain the indices of the objects that the corresponding object depends on."
    )
    wavelength: float = Field(gt=0, description="Wavelength of the wave, in millimeters.")
    samples_per_wavelength: int = Field(gt=0, description="Number of samples per wavelength for the simulation.")

    @model_validator(mode="after")
    def _check_dag(self) -> "Scene":
        n = len(self.objs)
        if len(self.dag) != n:
            raise ValueError(
                f"dag has {len(self.dag)} entries but there are {n} objects; it must have exactly one entry per object"
            )

        for dst, deps in enumerate(self.dag):
            for src in deps:
                if not 0 <= src < n:
                    raise ValueError(f"dag[{dst}] refers to object index {src}, which is out of range for {n} objects")
                if src == dst:
                    raise ValueError(f"dag[{dst}] depends on itself")
            if len(set(deps)) != len(deps):
                raise ValueError(f"dag[{dst}] lists the same dependency more than once")

        self._check_physics()
        # Also the acyclicity check: this raises on a cycle.
        self.topological_order()
        return self

    def _check_physics(self) -> None:
        """Enforce what each object type means for the flow of light."""
        for dst, deps in enumerate(self.dag):
            if self.objs[dst].type == "source" and deps:
                raise ValueError(
                    f"dag[{dst}] gives dependencies to object {dst}, which is a "
                    "source; sources emit light and cannot receive it"
                )
            for src in deps:
                if self.objs[src].type == "detector":
                    raise ValueError(
                        f"dag[{dst}] depends on object {src}, which is a detector; "
                        "detectors absorb light and cannot re-radiate it"
                    )

    def topological_order(self) -> tuple[int, ...]:
        """Object indices ordered so every dependency precedes its dependents.

        `simulate` must propagate in this order: an object's field is only
        complete once everything feeding into it has been computed. Iterating
        `objs` in list order instead would silently produce zeros wherever a
        source happens to be listed after its destination.

        Raises:
            ValueError: if the graph contains a cycle.
        """
        n = len(self.objs)
        # Kahn's algorithm, iterative so a long chain cannot overflow a stack.
        remaining_deps = [len(deps) for deps in self.dag]
        dependents: list[list[int]] = [[] for _ in range(n)]
        for dst, deps in enumerate(self.dag):
            for src in deps:
                dependents[src].append(dst)

        ready = [i for i in range(n) if remaining_deps[i] == 0]
        order: list[int] = []
        while ready:
            node = ready.pop()
            order.append(node)
            for dst in dependents[node]:
                remaining_deps[dst] -= 1
                if remaining_deps[dst] == 0:
                    ready.append(dst)

        if len(order) != n:
            unresolved = [i for i in range(n) if remaining_deps[i] > 0]
            raise ValueError(f"dag contains a cycle involving object indices {unresolved}; the graph must be acyclic")
        return tuple(order)

    def sample_counts(self) -> list[int]:
        """Element count for each object, at this scene's sampling density."""
        return [obj.geometry.sample_count(self.samples_per_wavelength, self.wavelength) for obj in self.objs]

    def total_sample_count(self) -> int:
        """Total number of boundary elements this scene discretises into."""
        return sum(self.sample_counts())

    def propagation_pair_count(self) -> int:
        """Source-destination element pairs evaluated across the whole scene.

        This is the quantity that actually drives runtime: each DAG edge costs
        n_src * n_dst kernel evaluations. It grows quadratically with sampling
        density, so it is a far better cost estimate than the element count.
        """
        counts = self.sample_counts()
        return sum(counts[src] * counts[dst] for dst, deps in enumerate(self.dag) for src in deps)
