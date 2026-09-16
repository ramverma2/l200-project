# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenTelemetry Distributed Tracing with Automatic PII Redaction."""

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.pii_redactor import redact_pii

logger = logging.getLogger("blackjack_telemetry")
tracer = trace.get_tracer("blackjack_tutor", "1.0.0")


def _flatten_and_sanitize_attributes(prefix: str, data: Any) -> dict[str, Any]:
    """Flattens nested dicts and scrubs PII for OpenTelemetry span attributes."""
    sanitized = redact_pii(data)
    flat: dict[str, Any] = {}

    if isinstance(sanitized, dict):
        for k, v in sanitized.items():
            attr_key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (str, int, float, bool)):
                flat[attr_key] = v
            else:
                flat[attr_key] = str(v)
    elif isinstance(sanitized, (str, int, float, bool)):
        flat[prefix] = sanitized
    else:
        flat[prefix] = str(sanitized)

    return flat


@contextlib.contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]:
    """Context manager for an OpenTelemetry distributed tracing span with PII redaction.

    Args:
        name: Name of the distributed trace span (e.g. 'blackjack.check_strategy').
        attributes: Dictionary of attributes to attach to the span.

    Yields:
        The active OpenTelemetry Span object.
    """
    with tracer.start_as_current_span(name) as span:
        if attributes:
            sanitized_attrs = _flatten_and_sanitize_attributes("", attributes)
            for key, val in sanitized_attrs.items():
                span.set_attribute(key, val)
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def add_span_event(
    span: trace.Span,
    event_name: str,
    event_data: dict[str, Any] | None = None,
) -> None:
    """Adds a named event to an OpenTelemetry span, scrubbing PII.

    Args:
        span: Active OpenTelemetry span.
        event_name: Name of the span event.
        event_data: Event attributes.
    """
    attributes = _flatten_and_sanitize_attributes("", event_data or {})
    span.add_event(name=event_name, attributes=attributes)
