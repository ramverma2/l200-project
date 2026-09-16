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

"""PII Redactor: Scrubs sensitive personal data from logging, telemetry, and memory pipelines."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

# Regular expression patterns for common PII
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_PATTERN = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
API_KEY_PATTERN = re.compile(r"\b(?:AIza[0-9A-Za-z-_]{20,}|sk-[a-zA-Z0-9]{20,})\b")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


class PIIRedactor:
    """Detects and redacts Personally Identifiable Information (PII) across strings and nested structures."""

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Redacts sensitive PII from a text string.

        Args:
            text: The raw text string.

        Returns:
            The sanitized string with sensitive patterns replaced by redaction markers.
        """
        if not isinstance(text, str):
            return text

        scrubbed = API_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)
        scrubbed = CREDIT_CARD_PATTERN.sub("[REDACTED_CREDIT_CARD]", scrubbed)
        scrubbed = SSN_PATTERN.sub("[REDACTED_SSN]", scrubbed)
        scrubbed = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", scrubbed)
        scrubbed = PHONE_PATTERN.sub("[REDACTED_PHONE]", scrubbed)
        scrubbed = IP_PATTERN.sub("[REDACTED_IP]", scrubbed)
        return scrubbed

    @classmethod
    def redact(cls, data: Any) -> Any:
        """Recursively traverses dictionaries, lists, and strings to sanitize PII.

        Args:
            data: Data of any type (dict, list, str, etc.)

        Returns:
            Sanitized copy of the input data.
        """
        if isinstance(data, str):
            return cls.redact_text(data)
        elif isinstance(data, Mapping):
            return {k: cls.redact(v) for k, v in data.items()}
        elif isinstance(data, Sequence) and not isinstance(data, (bytes, bytearray)):
            return [cls.redact(item) for item in data]
        return data


def redact_pii(data: Any) -> Any:
    """Convenience functional wrapper for PIIRedactor.redact."""
    return PIIRedactor.redact(data)
