"""Run the API with uvicorn: `python -m beamline_playground.server`."""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the beamline-playground REST API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper())

    import uvicorn

    uvicorn.run(
        "beamline_playground.server.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
