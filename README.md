# beamline-playground

2D wave-optics simulation of beamlines, built on [JAX](https://docs.jax.dev/).

Optical elements are described as curves in a plane. Each curve is discretised
into boundary elements, and a complex field is propagated from element to
element along a dependency graph using a Rayleigh-Sommerfeld diffraction
integral. The result is the field and intensity along every object, which is
enough to see interference, diffraction and focusing.

## Install

```sh
pip install beamline-playground            # library only
pip install 'beamline-playground[server]'  # plus the REST API
```

The `server` module is optional. Importing the library never requires FastAPI.

With nix, add `overlays.default` to your nixpkgs and take the package from
`python3Packages`:

```nix
pkgs.python3.withPackages (ps:
  [ ps.beamline-playground ]
  ++ ps.beamline-playground.optional-dependencies.server)  # extra is opt-in
```

Or just run the API: `nix run github:you/beamline_playground#server`.
See [workflow.md](workflow.md) for every flake output.

## Quick start

```python
from beamline_playground import Scene, Source, Slit, Detector, Segment, Vec2, simulate

scene = Scene(
    name="single slit",
    objs=[
        # A 4 mm aperture, uniformly illuminated.
        Source(geometry=Segment(pos_a=Vec2(x=0, y=-2), pos_b=Vec2(x=0, y=2))),
        # A screen 2 m downstream.
        Detector(geometry=Segment(pos_a=Vec2(x=2000, y=-600), pos_b=Vec2(x=2000, y=600))),
    ],
    # objs[0] has no inputs (it emits); objs[1] receives from objs[0].
    dag=[[], [0]],
    wavelength=0.5,
    samples_per_wavelength=4,
)

result = simulate(scene)
screen = result.objs[1]
print(max(screen.intensity_y))   # peak intensity
print(screen.sample_x[:3])       # position along the screen, in mm
print(screen.field[:3])          # complex field, as a numpy array
```

That scene reproduces the single-slit `sinc^2` pattern to within 0.1% RMS; see
`tests/test_diffraction.py`.

There is a runnable example in [`examples/basic_scene.py`](examples/basic_scene.py).

## Concepts

**Units.** Every length is in millimetres, including the wavelength. Angles
are in radians.

**Geometry.** `Segment`, `Circle`, `Arc`, `Ellipse` and `Parabola`. Each knows
its own arc length and how to discretise itself, so sample positions and
lengths can never disagree.

**Objects.** A geometry plus a role, and the role determines what happens to
the light:

| Type | Behaviour |
| --- | --- |
| `Source` | Emits at unit amplitude. Cannot receive. |
| `Mirror` | Reflects whatever reaches it. |
| `Slit` | Transmits whatever reaches it. |
| `Detector` | Absorbs. Terminal, so nothing may depend on it. |

**The DAG.** `scene.dag[i]` lists the indices of the objects that feed into
object `i`. It is validated on construction: lengths must match, indices must
be in range, and the graph must be acyclic and physically coherent.

**Sampling.** `samples_per_wavelength` sets the discretisation density. Samples
sit at *element midpoints*, so `dx` is exactly the element width, `sum(dx)`
equals the arc length, and a closed curve never double counts a point.
Cost grows quadratically with density: use `scene.propagation_pair_count()` to
check before running something large.

## Propagation kernels

Every kernel in `beamline_playground.algorithms` satisfies the same
`Propagator` interface, so they are interchangeable:

```python
from beamline_playground.simulate import NEAR_FIELD_PROPAGATORS
result = simulate(scene, propagators=NEAR_FIELD_PROPAGATORS)
```

- **`FAR_FIELD_PROPAGATORS`** (default) — the Rayleigh-Sommerfeld integral.
  Accurate when propagation distances are large compared to the wavelength.
  This is the path covered by the analytic tests.
- **`NEAR_FIELD_PROPAGATORS`** — Hankel-function boundary-element kernels,
  valid close to an object. **Experimental and not yet validated against
  analytic results.**

Memory is bounded: the destination axis is processed in chunks, so a full
`n_dst x n_src` matrix is never allocated. Tune with `max_matrix_entries`.

## Server

An optional REST API for a frontend or thin client.

```sh
# In the nix dev shell (the project is importable, not installed):
python -m beamline_playground.server --reload
uvicorn beamline_playground.server.app:create_app --factory --reload

# After `pip install`, a console script is also available:
beamline-server --reload
```

`beamline-server` comes from the package's entry point, so it exists only once
the project is installed. `nix develop` puts `src/` on `PYTHONPATH` rather than
installing, so use `python -m` there.

| Route | Purpose |
| --- | --- |
| `GET /api/v1/health` | Liveness and version. |
| `POST /api/v1/simulate` | Simulate a scene. |
| `POST /api/v1/scene/cost` | Estimate a scene's cost without running it. |

Interactive docs at `/docs`; the OpenAPI schema at `/openapi.json`.

Simulations run in a thread pool, so a long one does not block other requests.
Scenes above a configurable budget are rejected with `422` rather than being
allowed to exhaust the machine.

Configuration is environment driven:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BEAMLINE_CORS_ORIGINS` | common localhost dev ports | Comma-separated allowed origins. Empty disables CORS. |
| `BEAMLINE_MAX_PROPAGATION_PAIRS` | `50000000` | Largest accepted `propagation_pair_count()`. |
| `BEAMLINE_MAX_TOTAL_ELEMENTS` | `200000` | Largest accepted `total_sample_count()`. |

The defaults suit local development. **Review the CORS origins and the budget
before exposing this beyond localhost.**

## Development

A nix flake provides the dev shell and reads its dependencies from
`pyproject.toml`:

```sh
nix develop          # shell with every dependency, src/ on PYTHONPATH
```

Inside that shell:

```sh
pytest                                  # whole suite (~3 s)
pytest tests/test_diffraction.py -v     # one file, with test names
pytest -k fraunhofer                    # anything matching a keyword
pytest -x --lf                          # stop at first failure, then rerun just it

ruff check .                            # lint
ruff check --fix .                      # lint and autofix
mypy                                    # type-check

python examples/basic_scene.py          # run the example
python -m beamline_playground.server --reload   # serve the API
```

The shell provides dependencies but does not install the project; `src/` is on
`PYTHONPATH` instead, so edits take effect immediately with no reinstall step.
One consequence: `importlib.metadata` cannot see a version, so
`__version__` reads `0.0.0+unknown` until the package is actually installed.

To check the packaging itself:

```sh
nix build .#default  # build the wheel as a nix derivation
nix flake check      # evaluate every flake output
```

Both read only git-tracked files, so `git add` new files before running them.

Without nix, `pip install -e '.[server,dev]'` provides the same tooling.
