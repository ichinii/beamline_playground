from fastapi import FastAPI
from ..scene import Scene
from ..simulate import simulate, SimulationResult, ObjectResult
from ..instance import SceneInstance

app = FastAPI()

@app.post("/simulate")
def app_simulate(scene: Scene) -> SimulationResult:
    return simulate(scene)

# TODO: remove this
# test if everything works
from ..scene import Source, Mirror, Slit, Detector, Segment, Arc, Circle, Parabola, Vec2
result = app_simulate(Scene(name="Example Scene",
    objs=[
        Source(geometry=Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1))),
        # Mirror(geometry=Segment(pos_a=Vec2(x=1, y=-1), pos_b=Vec2(x=3, y=1))),
        # Mirror(geometry=Arc(pos=Vec2(x=1, y=-1), radius=2, angle=1, normal_inward=True)),
        # Mirror(geometry=Circle(pos=Vec2(x=1, y=-1), radius=2, normal_inward=True)),
        Mirror(geometry=Parabola(pos=Vec2(x=1, y=-1), focal_length=2, angle=1, normal_inward=True)),
        Slit(geometry=Segment(pos_a=Vec2(x=-2.1, y=2), pos_b=Vec2(x=2.1, y=2))),
        Detector(geometry=Segment(pos_a=Vec2(x=-1, y=3), pos_b=Vec2(x=1, y=3))),
    ],
    dag=[
        [],
        [0],
        [1],
        [2]
    ],
    wavelength=5,
    samples_per_wavelength=10),
    )
print(result)
