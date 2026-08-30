from __future__ import annotations

import os
from typing import Any

from . import __version__
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
            "The HTTP service requires the 'api' extra: pip install 'genmat[api]'"
        ) from exc

    runtime = service or GeneratorService(config or ServiceConfig.from_env())
    app = FastAPI(
        title="GenMat Generation API",
        version=__version__,
        description=(
            "Auditable crystal hypothesis generation with algorithmic, GenMat, "
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
            "name": "GenMat Generation API",
            "version": __version__,
            "health": "/v1/health",
            "capabilities": "/v1/capabilities",
            "models": "/v1/models",
            "generate": "/v1/generate",
            "documentation": "/docs",
        }

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return runtime.capabilities()

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return runtime.models()

    @app.get("/v1/models/{model_ref:path}")
    def model_info(model_ref: str) -> dict[str, Any]:
        try:
            return runtime.model_info(model_ref)
        except BackendUnavailableError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
            "The HTTP service requires the 'api' extra: pip install 'genmat[api]'"
        ) from exc
    if argv:
        raise ValueError(
            "genmat-api uses GENMAT_API_HOST, GENMAT_API_PORT, and "
            "GENMAT_API_LOG_LEVEL (GENIM_* aliases remain supported)"
        )
    host = os.environ.get("GENMAT_API_HOST", os.environ.get("GENIM_API_HOST", "127.0.0.1"))
    port = int(os.environ.get("GENMAT_API_PORT", os.environ.get("GENIM_API_PORT", "8000")))
    log_level = os.environ.get(
        "GENMAT_API_LOG_LEVEL", os.environ.get("GENIM_API_LOG_LEVEL", "info")
    )
    uvicorn.run(create_app(), host=host, port=port, log_level=log_level)
    return 0


__all__ = ["create_app", "main"]
