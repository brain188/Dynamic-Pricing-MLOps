"""Audit logging.

Every prediction attempt — success or failure — is appended as one JSON
line to `audit_log_path`. This is the system's source of truth for "what
did we predict, when, for what input, and did it work":

    * The `data_and_model_monitoring` pipeline reads
      this file to compute prediction drift, error rates, and latency
      trends over time.
    * An ops/monitoring dashboard can tail it directly to alert on
      `status == "failure"` in near real time.
    * It's the audit trail for "why did this ride cost this much" —
      every record includes the exact feature values used, not just the
      output.

Uses plain file I/O with a lock rather than a database, deliberately: an
audit log must never be the reason a request fails, and appending a JSON
line is about as close to "can't fail" as logging gets. For a
higher-throughput deployment, this will be swapped for an async queue (e.g. writing
to Kafka/SQS) without changing any caller — `AuditLogger.log()` is the
only integration point.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


class AuditLogger:
    """Appends structured audit records to a JSONL file."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        request_id: str,
        event_type: str,
        status: str,
        input_payload: Optional[Dict[str, Any]] = None,
        prediction: Optional[float] = None,
        model_version: Optional[str] = None,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        
        """Writes one audit record.

        Args:
            request_id: Correlates this record with request logs.
            event_type: e.g. "prediction", "startup", "health_check".
            status: "success" or "failure".
            input_payload: The validated request body, for full
                traceability of what produced a given prediction.
            prediction: The predicted fare, if successful.
            model_version: The MLflow registry version that served this
                request — critical for tracing a bad prediction back to a
                specific model after a later retrain.
            latency_ms: End-to-end request handling time.
            error: Error message, if status is "failure".
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "event_type": event_type,
            "status": status,
            "input": input_payload,
            "prediction": prediction,
            "model_version": model_version,
            "latency_ms": latency_ms,
            "error": error,
        }

        line = json.dumps(record)
        try:
            with _write_lock:
                with self.log_path.open("a") as f:
                    f.write(line + "\n")
        except OSError:
            # The audit log must never take the API down. Fall back to the
            # regular application logger so the failure is still visible
            # in container logs, then move on.
            logger.exception("Failed to write audit record for request_id=%s", request_id)
