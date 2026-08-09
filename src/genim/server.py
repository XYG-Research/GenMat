from __future__ import annotations

import os
from typing import Any

from .service import BackendUnavailableError, GeneratorService, ServiceConfig


def create_app(
    *,
    config: ServiceConfig | None = None,
    service: GeneratorService | None = None,
) -> Any:
    """Create the optional FastAPI application without burdening core imports."""

    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as exc:
        raise ImportError(
            "The HTTP service requires the 'api' extra: pip install 'genim[api]'"
        ) from exc

    runtime = service or GeneratorService(config or ServiceConfig.from_env())
    app = FastAPI(
        title="GenIM Generation API",
        version="0.3.0",
        description=(
            "Auditable crystal hypothesis generation with algorithmic, GenIM, "
            "Matra, and ensemble backends."
        ),
    )
    if runtime.config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime.config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["content-type"],
        )

    @app.get("/")
    def index() -> dict[str, Any]:
        return {
            "name": "GenIM Generation API",
            "version": "0.3.0",
            "health": "/v1/health",
            "capabilities": "/v1/capabilities",
            "generate": "/v1/generate",
            "documentation": "/docs",
        }

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return runtime.capabilities()

    @app.post("/v1/generate")
    def generate(payload: dict[str, Any] | None = Body(default=None)) -> dict[str, Any]:
        try:
            return runtime.generate(payload)
        except BackendUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> int:
    """Run the API with safe local defaults; environment variables override them."""

    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "The HTTP service requires the 'api' extra: pip install 'genim[api]'"
        ) from exc
    if argv:
        raise ValueError("genim-api uses GENIM_API_HOST, GENIM_API_PORT, and GENIM_API_LOG_LEVEL")
    host = os.environ.get("GENIM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("GENIM_API_PORT", "8000"))
    log_level = os.environ.get("GENIM_API_LOG_LEVEL", "info")
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level)
    return 0


__all__ = ["create_app", "main"]

