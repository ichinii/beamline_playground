# Development workflow

Every command here has been run and verified in the nix dev shell.

## Enter the shell once

```sh
nix develop
```

That gives you python 3.13.15 plus jax, numpy, scipy, pydantic, fastapi,
uvicorn, httpx, and `pytest` / `ruff` / `mypy` on `PATH`. Stay in that shell
for everything below.

**Important:** the shell provides *dependencies* but does not install the
project. `src/` goes on `PYTHONPATH` instead, so edits take effect immediately
with no reinstall step. Two consequences:

- `import beamline_playground` just works.
- `__version__` reads `0.0.0+unknown`, because `importlib.metadata` has no
  installed distribution to read. This is expected, not a fault — it also
  shows up in the API's `/health` response. It becomes a real version only
  once the package is actually installed.

Without nix, `pip install -e '.[server,dev]'` provides the same tooling.

## Tests

```sh
pytest                                # whole suite: 177 tests, ~3 s
pytest tests/test_diffraction.py -v   # one file, with test names
pytest -k fraunhofer                  # keyword match
pytest -x --lf                        # stop at first failure, then rerun just it
```

The suite is fast enough to run on every save.

| File | Covers |
| --- | --- |
| `tests/test_geometry.py` | Discretisation invariants: `sum(dx) == length`, unit normals, orientation, no duplicated endpoint on closed curves |
| `tests/test_diffraction.py` | Analytic physics: single-slit Fraunhofer `sinc^2`, cylindrical `1/r` decay, phase accumulation, linearity, convergence |
| `tests/test_scene_validation.py` | Rejecting bad input: dag shape, cycles, out-of-range indices, physics rules |
| `tests/test_simulate.py` | Pipeline: type dispatch, chunking equivalence, result serialisation |
| `tests/test_server.py` | REST API: routes, cost guard, CORS, config. Skips if the `server` extra is absent |

If you touch kernel or geometry maths, `test_diffraction.py` is the one that
matters. It compares against closed-form results rather than against the
current output, so it catches physics regressions instead of just crashes.

## Server

```sh
python -m beamline_playground.server --reload
```

Or drive uvicorn yourself:

```sh
uvicorn beamline_playground.server.app:create_app --factory --port 8000 --reload
```

`beamline-server` is a package entry point, so it exists **only after an
install**. It is not available in `nix develop` — use `python -m` there.

### Routes

| URL | Purpose |
| --- | --- |
| `GET /api/v1/health` | Liveness and version |
| `POST /api/v1/simulate` | Simulate a scene |
| `POST /api/v1/scene/cost` | Estimate a scene's cost without running it |
| `GET /openapi.json` | OpenAPI schema |
| `GET /docs` | Interactive Swagger docs |
| `GET /redoc` | ReDoc docs |

Note the schema and docs live at the **root**, not under `/api/v1`. This is
stock FastAPI: `APIRouter(prefix=...)` applies only to routes registered on
that router, while `/openapi.json`, `/docs` and `/redoc` are created by the
`FastAPI()` constructor and mounted on the app itself. So `/api/v1/openapi.json`
correctly 404s. They can be moved under the prefix by passing `openapi_url`,
`docs_url` and `redoc_url` to `FastAPI()` if you would rather everything the
API owns sit under one path.

### Quick manual check

```sh
curl -s localhost:8000/api/v1/health

curl -s -X POST localhost:8000/api/v1/simulate \
  -H 'Content-Type: application/json' -d '{
  "name": "demo",
  "objs": [
    {"type": "source",   "geometry": {"type": "segment", "pos_a": {"x": 0, "y": -1}, "pos_b": {"x": 0, "y": 1}}},
    {"type": "detector", "geometry": {"type": "segment", "pos_a": {"x": -1, "y": 3}, "pos_b": {"x": 1, "y": 3}}}
  ],
  "dag": [[], [0]],
  "wavelength": 5,
  "samples_per_wavelength": 10
}'
```

Hitting `/api/v1/scene/cost` with the same body first is a cheap way to see
how expensive a scene is before committing to it.

## Lint and types

```sh
ruff check .          # lint
ruff check --fix .    # lint and autofix
mypy                  # type-check
```

Both are clean. CI runs these plus the suite, so running them before pushing
saves a round trip.

## Running the example

```sh
python examples/basic_scene.py
```

## Packaging checks

```sh
nix flake check      # evaluate every flake output
nix build .#library  # build the library as a nix derivation
```

Both read **only git-tracked files**. New files are invisible to nix until
staged, and the failure modes differ confusingly: an untracked `pyproject.toml`
makes the flake fail to evaluate at all, while an untracked source module fails
*silently* and produces a broken package. Run `git add` on new files first.

See "Building and consuming the package with nix" below for the full set of
outputs.

## Building and consuming the package with nix

Nix has no equivalent of `pip install pkg[server]` — an extra cannot be
requested at install time, so each combination is its own output. The flake
exposes them explicitly.

### Outputs

| Output | What it is |
| --- | --- |
| `packages.library` (= `default`) | The library derivation. Core dependencies only. |
| `packages.python` | A python interpreter with the library importable. |
| `packages.python-with-server` | The same, plus the `server` extra. |
| `packages.server` | Wrapper that runs the REST API. |
| `apps.server` (= `default`) | `nix run .#server` |
| `overlays.default` | Adds `beamline-playground` to `python3Packages`. |
| `devShells.default` | `nix develop`, every extra, project not installed. |
| `packages.docker` | Container image for the server (Linux hosts only). |

```sh
nix build .#library              # just the library
nix build .#python-with-server   # ./result/bin/python has the server importable
nix run .#server -- --port 8000  # serve the API

nix build .#docker               # container image as a loadable tarball
docker load -i result
docker run -p 8000:8000 beamline-playground:latest
```

`packages.docker` is guarded to Linux hosts, since `dockerTools` cannot build
Linux images on darwin.

### Consuming it from another flake

This is the path for a user. Add the overlay, then pull the package out of
`python3Packages` like any other:

```nix
{
  inputs.beamline.url = "github:you/beamline_playground";

  outputs = { nixpkgs, beamline, ... }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        overlays = [ beamline.overlays.default ];
      };
    in {
      # library only
      packages.x86_64-linux.default =
        pkgs.python3.withPackages (ps: [ ps.beamline-playground ]);

      # library plus the server extra
      packages.x86_64-linux.withServer =
        pkgs.python3.withPackages (ps:
          [ ps.beamline-playground ]
          ++ ps.beamline-playground.optional-dependencies.server);
    };
}
```

`optional-dependencies` is the nixpkgs convention for extras: the derivation
carries them in `passthru`, and a consumer concatenates the ones they want into
their environment. `dev` is available the same way.

### Why there is no single "install with server" output

A python *library* derivation is not usable on its own — `nix shell .#library`
puts its `bin/` on `PATH` but gives you the system python, which knows nothing
about the package. A library has to be placed inside a `python.withPackages`
environment, which is what `packages.python*` are for.

Relatedly, the `beamline-server` console script inside `packages.library`
**cannot run**. `buildPythonPackage` wraps console scripts with the package's
core dependencies only, so the script starts and then fails on `import fastapi`
with the library's own "optional dependency" error. That is why
`packages.server` invokes `python -m beamline_playground.server` from the
server-enabled environment instead of reusing the entry point.

The console script is still correct for `pip` users, where
`pip install 'beamline-playground[server]'` does put FastAPI on the path.

## Dependency versions

The flake pins `nixos-26.05` (stable), which supplies python 3.13.15,
jax 0.10.0, numpy 2.4.4, scipy 1.17.1, pydantic 2.12.5, fastapi 0.136.3,
pytest 9.0.3, setuptools 80.10.1. Nothing in the project requires unstable;
`setuptools >= 77` (for the PEP 639 license expression) was the tightest
constraint and stable clears it comfortably.

Dependencies are declared **once**, in `pyproject.toml`. The flake reads them
from there via pyproject-nix, so there is no second list to keep in sync. To
add a dependency, edit `pyproject.toml` and re-enter `nix develop`.

## CI

Runs on pull requests, pushes to `main`, and manual dispatch.

| Job | Purpose |
| --- | --- |
| `test` | The suite on python 3.11, 3.12 and 3.13 |
| `lint` | ruff and mypy |
| `core-only` | Installs *without* the server extra, asserts FastAPI really is absent, and that the library still imports and the suite still passes |
| `example` | `examples/basic_scene.py` actually runs |
| `packaging` | Builds wheel and sdist, `twine check`, asserts `py.typed` is in the wheel |
| `nix` | `nix flake check`, builds the library and the server env, asserts the core env cannot import the server module |
| `docker` | Builds the image, runs the container, and checks health, correct physics against the analytic result, and that the cost guard returns 422 |

### Why most jobs use pip rather than nix

Nix pays off when it saves compilation. Every dependency here ships a prebuilt
wheel on PyPI, so there is nothing to compile, and nix would only add transfer:

| | Cold download |
| --- | --- |
| pip wheel set | ~171 MiB, plus `setup-python` pip caching |
| nix dev shell closure | ~421 MiB compressed, 1.5 GiB unpacked |

So pip drives the fast matrix jobs, and nix is used only where it is
structurally required: evaluating flake outputs and building the image. The
`docker` job is by far the slowest (it fetches a ~1 GiB closure); move it to
`main` only if PR latency matters more than catching image regressions early.
