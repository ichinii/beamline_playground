from typing import Literal, Union, Annotated, List
from pydantic import BaseModel, Field
from typing_extensions import Self
from uuid import uuid4
import numpy as np
import math

class Vec2(BaseModel):
    x: float = Field(description="X coordinate, in millimeters. In an unrotated context, the positive X axis points to the right.")
    y: float = Field(description="Y coordinate, in millimeters. In an unrotated context, the positive Y axis points upwards.")

def distance(a: Vec2, b: Vec2) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

# geometries / curvatures

class Segment(BaseModel):
    type: Literal["segment"] = "segment"
    pos_a: Vec2 = Field(description="The starting point of the segment, in millimeters. The normal vector is defined as the vector perpendicular to the segment, pointing to the right when looking from pos_a to pos_b.")
    pos_b: Vec2 = Field(description="The ending point of the segment in millimeters.")
    def length(self) -> float:
        return distance(self.pos_a, self.pos_b)

class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    pos: Vec2 = Field(description="The center of the circle, in millimeters.")
    radius: float = Field(gt=0, description="The radius of the circle, in millimeters.")
    normal_inward: bool = Field(description="If true, the normal vector points inward toward the center of the circle. If false, the normal vector points outward away from the center of the circle.")
    def length(self) -> float:
        return 2 * np.pi * self.radius

class Arc(BaseModel):
    type: Literal["arc"] = "arc"
    pos: Vec2 = Field(description="The center of the arc, in millimeters.")
    radius: float = Field(gt=0, description="The radius of the arc, in millimeters.")
    angle: float = Field(gt=0, le=2.0*math.pi, description="The angle of the arc, in radians. The center of the arc is at pos, and the arc extends from in both directions by angle/2.")
    normal_inward: bool = Field(description="If true, the normal vector points inward toward the center of the arc. If false, the normal vector points outward away from the center of the arc.")
    def length(self) -> float:
        return self.radius * self.angle

class Ellipse(BaseModel):
    type: Literal["ellipse"] = "ellipse"
    pos: Vec2 = Field(description="The center of the ellipse, in millimeters.")
    radius_x: float = Field(gt=0, description="The radius of the ellipse along the X axis, in millimeters.")
    radius_y: float = Field(gt=0, description="The radius of the ellipse along the Y axis, in millimeters.")
    angle: float = Field(gt=0, le=2.0*math.pi, description="The angle of the ellipse, in radians. The center of the ellipse is at pos, and the ellipse extends from in both directions by angle/2.")
    normal_inward: bool = Field(description="If true, the normal vector points inward toward the center of the ellipse. If false, the normal vector points outward away from the center of the ellipse.")
    def length(self) -> float:
        # Approximate length of an elliptical segment using numerical integration
        angles = np.linspace(0, self.angle, 100)
        x = self.pos.x + self.radius_x * np.cos(angles)
        y = self.pos.y + self.radius_y * np.sin(angles)
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx**2 + dy**2))

class Parabola(BaseModel):
    type: Literal["parabola"] = "parabola"
    pos: Vec2 = Field(description="The vertex of the parabola, in millimeters.")
    focal_length: float = Field(gt=0, description="The focal length of the parabola, in millimeters.")
    angle: float = Field(gt=0, le=2.0*math.pi, description="The angle of the parabola, in radians. The vertex of the parabola is at pos, and the parabola extends from in both directions by angle/2.")
    normal_inward: bool = Field(description="If true, the normal vector points inward toward the center of the parabola. If false, the normal vector points outward away from the center of the parabola.")
    def length(self) -> float:
        # Approximate length of a parabolic segment using numerical integration
        angles = np.linspace(-self.angle/2, self.angle/2, 100)
        x = self.pos.x + angles
        y = self.pos.y + (x - self.pos.x)**2 / (4 * self.focal_length)
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx**2 + dy**2))

Geometry = Annotated[Union[Segment, Circle, Arc, Ellipse, Parabola], Field(discriminator="type")]

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
    geometry: Segment = Field(description="Geometry of the object. Always a segment, since slits are always straight lines.")

Object = Annotated[Union[Source, Detector, Mirror, Slit], Field(discriminator="type")]

class Scene(BaseModel):
    name: str = Field(default="<unnamed scene>", description="Name for the scene.")
    objs: list[Object] = Field(description="List of objects in the scene.")
    dag: list[list[int]] = Field(description="Directed acyclic graph (DAG) representing the relationships between objects in the scene. Each index in the outer list corresponds to an object in the objs list, and the inner lists contain the indices of the objects that the corresponding object depends on.")
    wavelength: float = Field(gt=0, description="Wavelength of the wave, in millimeters.")
    samples_per_wavelength: int = Field(gt=0, description="Number of samples per wavelength for the simulation.")
