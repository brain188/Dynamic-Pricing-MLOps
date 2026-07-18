"""Dynamic pricing FastAPI backend.

Serves real-time fare predictions from the MLflow-registered
`"production"`-aliased model. Every request — success or failure — is
audit-logged (see `audit.py`) so the monitoring pipeline and an ops
dashboard have a complete, queryable record of system behavior.

Endpoints:
    GET  /health   Liveness + model status.
    POST /predict  Score one ride request.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .audit import AuditLogger
from .config import get_settings
from .feature_pipeline import transform_request
from .logging_config import configure_logging
from .model_loader import load_model_bundle
from .schemas import ErrorResponse, HealthResponse, PredictionResponse, RideRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the model bundle once at startup; fails fast if it can't."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app.state.settings = settings
    app.state.audit = AuditLogger(settings.audit_log_path)

    try:
        app.state.model_bundle = load_model_bundle(settings)
        app.state.audit.log(
            request_id="startup",
            event_type="startup",
            status="success",
            model_version=app.state.model_bundle.model_version,
        )
    except Exception as exc:
        # A backend that can't load its model is not a valid degraded
        # state for a pricing service — fail the deploy loudly rather
        # than serve predictions from nothing.
        logger.exception("Fatal: model bundle failed to load at startup.")
        app.state.audit.log(
            request_id="startup", event_type="startup", status="failure", error=str(exc)
        )
        raise

    yield


app = FastAPI(title="Dynamic Pricing API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """Attaches a request_id to every request and logs total latency."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        extra={"request_id": request_id},
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Invalid request bodies: 422, audited, no stack trace leaked."""
    request_id = getattr(request.state, "request_id", "unknown")
    detail = str(exc.errors())
    request.app.state.audit.log(
        request_id=request_id, event_type="prediction", status="failure", error=detail
    )
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            request_id=request_id, error="validation_error", detail=detail
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Anything unanticipated: 500, full traceback logged server-side and
    audited, generic message returned to the caller."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception.", extra={"request_id": request_id})
    request.app.state.audit.log(
        request_id=request_id, event_type="prediction", status="failure", error=str(exc)
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            request_id=request_id,
            error="internal_error",
            detail="An unexpected error occurred. This has been logged for investigation.",
        ).model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    bundle = getattr(request.app.state, "model_bundle", None)
    settings = request.app.state.settings
    return HealthResponse(
        status="ok" if bundle else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.model_version if bundle else None,
        production_alias=settings.production_alias,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, ride_request: RideRequest) -> PredictionResponse:
    request_id = request.state.request_id
    start = time.perf_counter()

    bundle = request.app.state.model_bundle
    settings = request.app.state.settings

    encoded = transform_request(ride_request, bundle, settings)
    raw_prediction = float(bundle.model.predict(encoded)[0])

    # clip to sane business bounds rather than return an
    # implausible fare.
    is_out_of_bounds = not (settings.prediction_min <= raw_prediction <= settings.prediction_max)
    predicted_fare = min(max(raw_prediction, settings.prediction_min), settings.prediction_max)

    latency_ms = (time.perf_counter() - start) * 1000

    request.app.state.audit.log(
        request_id=request_id,
        event_type="prediction",
        status="success",
        input_payload=ride_request.model_dump(),
        prediction=predicted_fare,
        model_version=bundle.model_version,
        latency_ms=latency_ms,
    )

    return PredictionResponse(
        request_id=request_id,
        predicted_fare=predicted_fare,
        is_out_of_bounds=is_out_of_bounds,
        model_version=bundle.model_version,
        latency_ms=latency_ms,
    )
