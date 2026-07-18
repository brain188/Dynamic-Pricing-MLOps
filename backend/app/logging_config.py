"""Structured JSON logging.

Logs to stdout as one JSON object per line — the standard shape expected
by container log collectors (CloudWatch, Datadog, Loki, etc.), so ops
tooling can filter/alert on `level`, `request_id`, or `logger` without a
separate log parser.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # request_id is attached via logging.LoggerAdapter in main.py's
        # middleware, so every request-scoped log line is correlatable.
        if hasattr(record, "request_id"):
            payload["request_id"] = record.request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())

    # Quiet noisy third-party loggers down to warnings only.
    for noisy_logger in ("uvicorn.access", "mlflow"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
