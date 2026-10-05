"""Minimal end-to-end example: source -> parabolic mirror -> slit -> detector.

Run with:  python examples/basic_scene.py
"""

from beamline_playground import (
    Detector,
    Mirror,
    Parabola,
    Scene,
    Segment,
    Slit,
    Source,
    Vec2,
    simulate,
)


def build_scene() -> Scene:
    return Scene(
        name="Example Scene",
        objs=[
            Source(geometry=Segment(pos_a=Vec2(x=0, y=-1), pos_b=Vec2(x=0, y=1))),
            Mirror(geometry=Parabola(pos=Vec2(x=1, y=-1), focal_length=2, extent=1, normal_inward=True)),
            Slit(geometry=Segment(pos_a=Vec2(x=-2.1, y=2), pos_b=Vec2(x=2.1, y=2))),
            Detector(geometry=Segment(pos_a=Vec2(x=-1, y=3), pos_b=Vec2(x=1, y=3))),
        ],
        # Each entry lists the objects that feed into the object at that index.
        dag=[[], [0], [1], [2]],
        wavelength=5,
        samples_per_wavelength=10,
    )


def main() -> None:
    result = simulate(build_scene())
    for obj in result.objs:
        peak = max(obj.intensity_y) if obj.intensity_y else 0.0
        print(f"{obj.type:9s} {obj.name:20s} samples={len(obj.intensity_y):5d} peak_intensity={peak:.6g}")


if __name__ == "__main__":
    main()
