"""REST API over the simulation library.

Built as a factory (`create_app`) so that configuration, CORS and testing are
all straightforward. A module-level `app` is also provided for the common
`uvicorn ...:app` invocation.
"""

import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .. import __version__
from ..scene import Scene
from ..simulate import SimulationResult, simulate
from .config import ServerConfig

logger = logging.getLogger(__name__)

API_PREFIX = "/api/v1"


class HealthResponse(BaseModel):
    status: str = Field(description="Always 'ok' when the service is serving requests.")
    version: str = Field(description="Version of the beamline-playground package.")


class SceneCost(BaseModel):
    """What a scene would cost to simulate, and what the limits are."""

    total_elements: int = Field(description="Boundary elements the scene discretises into.")
    propagation_pairs: int = Field(description="Source-destination element pairs to evaluate.")
    max_total_elements: int = Field(description="Server limit on total_elements.")
    max_propagation_pairs: int = Field(description="Server limit on propagation_pairs.")
    within_limits: bool = Field(description="Whether this scene would be accepted by /simulate.")


def _measure(scene: Scene, config: ServerConfig) -> SceneCost:
    elements = scene.total_sample_count()
    pairs = scene.propagation_pair_count()
    return SceneCost(
        total_elements=elements,
        propagation_pairs=pairs,
        max_total_elements=config.max_total_elements,
        max_propagation_pairs=config.max_propagation_pairs,
        within_limits=(elements <= config.max_total_elements and pairs <= config.max_propagation_pairs),
    )


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: settings to use. Defaults to `ServerConfig.from_env()`.
    """
    config = ServerConfig.from_env() if config is None else config

    app = FastAPI(
        title="Beamline Playground API",
        version=__version__,
        description=(
            "Simulate 2D wave-optics beamlines. POST a scene to "
            f"{API_PREFIX}/simulate to get the field and intensity along every object."
        ),
    )

    # A browser frontend is the intended client, so cross-origin requests must
    # be permitted explicitly. Configure via BEAMLINE_CORS_ORIGINS.
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @router.post("/scene/cost", response_model=SceneCost, tags=["simulation"])
    def scene_cost(scene: Scene) -> SceneCost:
        """Estimate a scene's cost without running it.

        Lets a client check a scene against the server's limits before
        committing to a long request.
        """
        return _measure(scene, config)

    @router.post("/simulate", response_model=SimulationResult, tags=["simulation"])
    async def app_simulate(scene: Scene) -> SimulationResult:
        cost = _measure(scene, config)
        if not cost.within_limits:
            # 422 rather than 413: the payload is small, it is the work it
            # describes that is too large.
            raise HTTPException(
                # Literal rather than fastapi.status: the constant for 422 was
                # renamed between versions.
                status_code=422,
                detail={
                    "message": (
                        "scene exceeds this server's simulation budget; reduce "
                        "samples_per_wavelength or the size of the objects"
                    ),
                    "cost": cost.model_dump(),
                },
            )

        logger.info(
            "simulating %r (%d elements, %d pairs)",
            scene.name,
            cost.total_elements,
            cost.propagation_pairs,
        )
        # simulate() is synchronous and CPU-bound; running it inline would
        # block the event loop and stall every other request.
        return await run_in_threadpool(simulate, scene)

    app.include_router(router)
    return app


app = create_app()
