import math

import pytest

from beamline_playground import Arc, Circle, Ellipse, Parabola, Segment, Vec2


def segment(y: float = 0.0, half_width: float = 1.0) -> Segment:
    """A horizontal segment at height `y`."""
    return Segment(pos_a=Vec2(x=-half_width, y=y), pos_b=Vec2(x=half_width, y=y))


# One instance of every geometry, for tests that must hold for all of them.
ALL_GEOMETRIES = {
    "segment": Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1)),
    "circle": Circle(pos=Vec2(x=0, y=0), radius=2, normal_inward=True),
    "arc": Arc(pos=Vec2(x=0, y=0), radius=2, angle=1.0, normal_inward=True),
    "ellipse": Ellipse(pos=Vec2(x=0, y=0), radius_x=3, radius_y=1, angle=2.0, normal_inward=True),
    "parabola": Parabola(pos=Vec2(x=0, y=0), focal_length=2, extent=4, normal_inward=True),
    "closed_circle": Circle(pos=Vec2(x=1, y=1), radius=0.5, normal_inward=False),
    "full_arc": Arc(pos=Vec2(x=0, y=0), radius=1, angle=2 * math.pi, normal_inward=False),
}


@pytest.fixture(params=sorted(ALL_GEOMETRIES), ids=sorted(ALL_GEOMETRIES))
def geometry(request):
    return ALL_GEOMETRIES[request.param]
