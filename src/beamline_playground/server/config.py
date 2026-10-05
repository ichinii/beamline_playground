"""Server configuration, read from the environment."""

import os
from dataclasses import dataclass

# Vite, CRA and the common static-server ports, for local frontend development.
_DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
)

# Refuse scenes above this many source-destination element pairs. The default
# is a few seconds of work; raise it deliberately if you trust your callers.
_DEFAULT_MAX_PAIRS = 500_000_000

_DEFAULT_MAX_ELEMENTS = 200_000


@dataclass(frozen=True)
class ServerConfig:
    """Knobs for the REST API.

    Every field can be overridden by an environment variable so that a
    deployment needs no code change.
    """

    #: Browser origins allowed to call the API. `["*"]` disables the check.
    cors_origins: tuple[str, ...] = _DEFAULT_CORS_ORIGINS
    #: Largest accepted `Scene.propagation_pair_count()`.
    max_propagation_pairs: int = _DEFAULT_MAX_PAIRS
    #: Largest accepted `Scene.total_sample_count()`.
    max_total_elements: int = _DEFAULT_MAX_ELEMENTS

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ServerConfig":
        source = os.environ if env is None else env

        origins_raw = source.get("BEAMLINE_CORS_ORIGINS")
        if origins_raw is None:
            origins = _DEFAULT_CORS_ORIGINS
        else:
            origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip())

        return cls(
            cors_origins=origins,
            max_propagation_pairs=int(source.get("BEAMLINE_MAX_PROPAGATION_PAIRS", _DEFAULT_MAX_PAIRS)),
            max_total_elements=int(source.get("BEAMLINE_MAX_TOTAL_ELEMENTS", _DEFAULT_MAX_ELEMENTS)),
        )
