"""Optional REST API for a frontend or thin client.

Requires the `server` extra:

    pip install 'beamline-playground[server]'
"""

try:
    import fastapi as _fastapi
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "beamline_playground.server needs FastAPI, which is an optional "
        "dependency. Install it with: pip install 'beamline-playground[server]'"
    ) from exc

from .app import API_PREFIX, create_app
from .config import ServerConfig

__all__ = [
    "API_PREFIX",
    "ServerConfig",
    "create_app",
]
