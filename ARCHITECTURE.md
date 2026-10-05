# Architecture

## The one-paragraph version

A beamline is described as a set of curves in a 2D plane, plus a graph saying
which curve illuminates which. Each curve is chopped into small straight
*boundary elements*. A complex field is then pushed from element to element
along the graph by evaluating a diffraction integral, and what falls out is the
field and intensity along every object. Everything else in this repository
exists to make that one computation describable, fast, correct, or reachable
over HTTP.

## The pipeline

Three stages, deliberately separated by *what varies*:

```
   Scene                 SceneInstance              SimulationResult
 (declarative)          (discretised)                 (measured)

 "a 4mm slit at      "that slit is 32 elements      "here is the field
  x=0, lambda=0.5"    at these coordinates,          at each of those
                      with these normals"            32 positions"

   scene.py      ->      instance.py        ->        simulate.py
 resolution-free      resolution-fixed            + algorithms/
    pure data          jax arrays                   the kernels
```

- A `Scene` is resolution independent. It is also the wire format: it is what a
  client POSTs. Nothing in it depends on how finely you intend to sample.
- A `SceneInstance` pins the scene to one sampling density. This is where
  geometry becomes numbers on a device.
- `simulate` walks the graph, calls kernels, and converts the result back into
  plain JSON-friendly data.

The separation matters because the three stages change for different reasons.
You edit a `Scene` when the optics change; you rebuild a `SceneInstance` when
you want more accuracy; you touch `algorithms/` when the physics is wrong.

## Modules

### `scene.py` — the schema *and* the geometry

The largest module, and the only one that is two things at once. It holds:

1. **The pydantic models**: `Vec2`, the five geometries (`Segment`, `Arc`,
   `Circle`, `Ellispoid`, `Parabola`), the four object types, and `Scene`.
   This is simultaneously the Python API, the JSON schema, and the OpenAPI
   documentation.
2. **The discretisation maths**: each geometry knows its own `length()` and its
   own `sample(n)`.

**The central contract** is `GeometrySamples`, returned by `sample(n)`:

| Field | Meaning |
| --- | --- |
| `pos_x`, `pos_y` | element midpoint coordinates |
| `normal_x`, `normal_y` | unit normal at each midpoint |
| `dx` | element width; `sum(dx)` equals the arc length |
| `s` | arc-length coordinate from the middle, for plotting |

Two invariants in there are load-bearing:

- **`sum(dx) == length()`.** `dx` is a quadrature weight. If it does not sum to
  the arc length, every integral is mis-scaled.
- **Normals are unit vectors.** They feed the obliquity factor `cos(theta)`
  directly. A non-normalised normal silently scales the field.

**Why midpoints and not endpoints.** Sampling `linspace(a, b, n)` gives `n`
points with spacing `L/(n-1)`, endpoints included, which forces a choice about
what width the two end samples represent. Sampling element *midpoints* makes
`dx = L/n` exactly correct with no special cases, and on a closed curve it also
stops the first and last sample landing on the same physical point.

`Scene` additionally carries a `@model_validator` enforcing that the DAG is
well-formed (shape, index range, no self-loops, no duplicate edges, acyclic)
and physically coherent (sources cannot receive; nothing can depend on a
detector). This runs at construction, so the server rejects nonsense with a
`422` instead of raising an `IndexError` deep inside a jit trace.

### `instance.py` — the bridge to the device

Deliberately thin, and contains **no geometry maths at all**. For each object it
asks `sample_count()` how many elements are needed, calls `sample()`, and moves
the arrays onto the device via `jnp.asarray`.

Two choices worth knowing:

- `objs` is a list of dicts holding only what a kernel consumes (`pos_x`,
  `pos_y`, `normal_x`, `normal_y`, `dx`). It is a jax pytree, so jit can trace
  through it.
- `sample_coords` (the `s` array) is kept *outside* `objs` and stays as numpy.
  It is output metadata, never a kernel input, and keeping it out means it is
  never traced into the compiled graph.
- `dag` becomes nested **tuples**, because it is passed as a static jit
  argument and must be hashable.

### `simulate.py` — the core engine

This is the engine. `_simulate_impl` is the whole thing:

```python
fields = [initial_field(i) for i in range(len(dag))]   # sources bright, rest dark

for dst_index in order:                      # topological, NOT list order
    for src_index in dag[dst_index]:
        fields[dst_index] += propagators[src_index](
            k, fields[src_index], objs[src_index], objs[dst_index])
```

That is it. Every object gets a field array; sources start at unit amplitude and
everything else starts at zero; then each graph edge adds one object's
contribution to another's.

**The key insight: the Python loop is unrolled at trace time.** jit traces this
once and bakes the whole propagation graph into a single static XLA graph. There
is no loop at runtime. This is why `dag`, `is_source` and `propagators` must be
*static* jit arguments — their values determine the graph's structure, so they
have to be known while tracing.

`k` is the exception: it is traced, not static. That avoids a recompile when the
wavelength changes without changing any element count. But be aware that a
wavelength change big enough to change `sample_count` changes the array shapes,
and jit retraces on shapes regardless.

**Ordering is load-bearing.** The outer loop follows
`Scene.topological_order()`, not the order objects appear in `objs`. This is not
cosmetic: iterating in list order means an object listed *before* its
illuminator gets evaluated while still dark, and its field stays zero. That
produced a silent wrong answer — a detector reading exactly `0` for a perfectly
valid scene, purely because the mirror was listed after it. There is no error,
no warning, just a zero.

The order comes from a Kahn sort in `scene.py`, which doubles as the acyclicity
check: if it cannot order every object, there is a cycle. `tests/test_simulate.py`
asserts that shuffling the object list does not change the result.

**Type dispatch.** The kernel is chosen by the *source* object's type, so a
mirror reflects and a slit transmits. Detectors get `None` because the validator
guarantees nothing depends on them. Two maps are supplied:

- `FAR_FIELD_PROPAGATORS` (default) — Rayleigh-Sommerfeld everywhere. The
  validated path, covered by the analytic tests.
- `NEAR_FIELD_PROPAGATORS` — the Hankel kernels. Valid close to an object,
  **experimental and not validated against analytic results.**

**Results.** `ObjectResult` splits the complex field into `field_re` and
`field_im` float arrays. pydantic serialises `complex` as a *string*
(`{"field_y": ["1+2j"]}`), which a JS client would have to parse by hand. A
`.field` property reassembles the numpy complex array for Python callers.

### `algorithms/` — the interchangeable kernels

#### `base.py`

Defines the `Propagator` protocol, which is the seam that makes kernels
swappable:

```python
propagator(k, src_field, src_obj, dst_obj) -> jax.Array
```

Also `chunk_size_for()`, which converts a memory budget in matrix entries into
a destination chunk size.

#### `rayleigh_sommerfeld.py` — the default kernel

The RS1 diffraction integral. For each destination element it sums over all
source elements:

```
contribution = exp(i*k*r) * (1/sqrt(r)) * cos(theta) * src_field * dx
```

`exp(i*k*r)` is the propagation phase, `1/sqrt(r)` is 2D cylindrical spreading,
`cos(theta)` is the obliquity factor (how edge-on the source radiates toward
this destination), `dx` is the quadrature weight.

This is a *far-field* form — it omits the Hankel term that matters near an
object — so it is accurate when distances are large compared to the wavelength.
It reproduces single-slit Fraunhofer to 0.0012 RMS, which is what pins it.

**Chunking.** The naive implementation materialises a dense `n_dst x n_src`
complex matrix, which OOMs at realistic sampling. Instead `jax.lax.map` with
`batch_size` processes destinations in chunks: vectorised within a chunk,
sequential across chunks. Peak memory is bounded by `max_matrix_entries`
(default ~32 MB) no matter how fine the source sampling is. Tested identical to
the unchunked result.

#### `hankel.py` — near-field, experimental

The 2D free-space Green's function is `(i/4) H0(kr)`, so a proper
boundary-element formulation needs Hankel functions. jax has none, so these are
computed on the host by `scipy.special.hankel1` behind a `jax.pure_callback`,
parallelised across CPU cores with a thread pool.

Exposes `propagate_mirror` (single-layer) and `propagate_slit` (double-layer),
both conforming to `Propagator`. A hypersingular kernel is present but unused
and explicitly unvalidated — it has an untreated `1/r^2` singularity when two
elements coincide.

Chunking here is a host-side Python loop rather than `lax.map`, because
`pure_callback` does not vectorise cleanly under vmap.

### `server/` — the optional REST layer

Entirely optional. The library never imports it, and `import
beamline_playground` does not require FastAPI.

- **`app.py`** — `create_app()` factory. Three routes under
  `/api/v1`: `health`, `simulate`, `scene/cost`. Simulation runs via
  `run_in_threadpool` because it is synchronous and CPU-bound; inline it would
  block the event loop and stall every other request. Over-budget scenes are
  rejected `422` before any work starts.
- **`config.py`** — frozen dataclass read from environment
  variables: CORS origins and the two cost ceilings.
- **`__init__.py`** — raises a `ModuleNotFoundError` naming the install command
  if FastAPI is missing, rather than letting an opaque import error escape.
- **`__main__.py`** — `python -m beamline_playground.server`.

**The cost guard** is the architecturally interesting part. Runtime is driven by
`Scene.propagation_pair_count()` — the sum of `n_src * n_dst` over graph edges —
which grows *quadratically* with sampling density. A single small POST can
therefore request an unbounded amount of work. The server estimates before
running and refuses above a configurable budget. `scene/cost` exposes the same
estimate so a client can check first.

## Why each dependency exists

### Runtime

| Dependency | Why |
| --- | --- |
| **jax** | The kernels are dense array maths over large matrices. jax gives jit compilation (the whole propagation graph becomes one XLA graph), vectorisation, and GPU portability without rewriting. `lax.map` with `batch_size` provides the chunking that bounds memory. |
| **numpy** | Used for *host-side* work: geometry discretisation and the output arrays. Geometry setup happens once, off the hot path, where numpy is simpler and keeps `sample_coords` off the device entirely. The split is intentional: numpy before the device boundary, jax after. |
| **pydantic** | Does three jobs at once. It validates untrusted input at the boundary (the DAG validator), generates the JSON schema the frontend consumes, and documents every field via `Field(description=...)` which flows straight into OpenAPI. Writing this by hand would mean three parallel definitions that drift. |
| **scipy** | Only for `scipy.special.hankel1` in `hankel.py`. jax has no Hankel function. This is the sole reason scipy is a dependency. |

### Server extra

| Dependency | Why |
| --- | --- |
| **fastapi** | Reads the pydantic models directly, so routes, validation and OpenAPI docs come from the schema already written. `run_in_threadpool` solves the blocking problem. |
| **uvicorn** | ASGI server to actually run it. |

Both are an *extra* rather than a core dependency so the library stays usable
without them. CI has a dedicated `core-only` job asserting that.

### Dev extra

| Tool | Why |
| --- | --- |
| **pytest** | The analytic tests are the real safety net: they compare against closed-form `sinc^2` and `1/r` rather than against current output, so they catch physics regressions, not just crashes. |
| **httpx** | Required by `fastapi`'s `TestClient` (via starlette). Easy to miss, because an environment that happens to have it masks the omission. |
| **ruff** | Lint and import sorting. Caught a pile of dead imports left over from earlier refactors. |
| **mypy** | Type checking. Caught a real bug: a CORS default inferred as a fixed-length 6-tuple rather than `tuple[str, ...]`. |

### Build and environment

| Tool | Why |
| --- | --- |
| **setuptools** (>= 77) | Build backend. The floor is for PEP 639 `license = "MIT"`. |
| **pyproject.toml** | Single source of truth for dependencies, extras, and all tool config. |
| **nix flake** | Reproducible dev environment and build. Pinned to `nixos-26.05` (stable) — nothing requires unstable. |
| **pyproject-nix** | Lets the flake *read* dependencies from `pyproject.toml` instead of duplicating them in Nix. Without it the dep list would exist twice and drift. |

## Design decisions worth knowing

**Geometry owns its own maths.** Adding a geometry means implementing
`length()` and `sample(n)` on one class. Nothing else needs to change —
`instance.py` and the kernels are generic over `GeometrySamples`.

**The graph is static, the data is dynamic.** Scene topology is baked into the
compiled graph; field values flow through it. This is why the DAG is a static
jit argument and why changing topology costs a recompile.

**Validation at the boundary, not in the engine.** The engine assumes a
well-formed scene. Everything that could be malformed is caught by the pydantic
validator, so no kernel contains defensive checks.

**Extras are genuinely optional.** `server` is not imported by the library, is
not a core dependency, and is covered by a CI job that installs without it.

## Known gaps

- `NEAR_FIELD_PROPAGATORS` is unvalidated against analytic results. Only the
  far-field path is pinned by tests. The cheapest fix is a differential test:
  in a regime where near- and far-field must agree (distance >> wavelength),
  assert the Hankel kernels match Rayleigh-Sommerfeld, using the validated path
  as the oracle.
- The `sinc^2` benchmark cannot verify the obliquity factor, because the
  textbook Fraunhofer formula omits it too — deleting `cos(theta)` from the
  kernel *improves* agreement with `sinc^2`. `TestObliquityFactor` covers it
  separately. Treat this as a warning about analytic benchmarks generally:
  check what approximations your reference formula also makes.
- The hypersingular kernel in `hankel.py` is unused and has an untreated
  `1/r^2` singularity.
- `Mirror` and `Slit` currently differ only in which kernel they select, not in
  any boundary condition. There is no actual reflection coefficient or
  transmission function.
- No GPU testing. jax would use one, but nothing has been measured there.
